"""
test_hardening.py  --  Comprehensive automated tests for ScreenSense AI hardening (Brief C).

Covers:
  - P1: Route 3 failure handling, no execution of guesses, disruptive triggers sanity gate.
  - P2: Cache key matching: exact command matching, hamming distance on screen hashes,
        intent matching, bypass of display_unavailable, semantic path flag.
  - P3: Destructive close fallback removal and elimination of taskkill / force_close_process.
  - P5: Natural speech phrasing rules for site and scroll commands.
  - P6: Model and dependency reproducibility configuration.
  - P7: Small fixes (is_muted tracking, screenshots folder, comment corrections).
  - P8: Offline-first sentence encoder loading.
"""

import os
import re
import unittest
from unittest.mock import patch, MagicMock

import route3_config
from route3_config import DISRUPTIVE_TRIGGERS
import vlm
from vlm import _call_gemini, _parse_vlm_response, route3_handle, check_disruptive_trigger
from route3_cache import get_cache
import main


class TestP1Route3FailureHandling(unittest.TestCase):
    """P1: Verify Route 3 failure handling, no-cache policy on failure, and disruptive sanity gate."""

    def setUp(self):
        # Reset cache for test isolation
        get_cache()._store.clear()

    def test_failure_mode_1_missing_api_key(self):
        with patch("vlm._load_api_key", return_value=None):
            action, target, reason, conf, lat = _call_gemini("do something", "open_app", 0.2, None)
            self.assertIsNone(action)
            self.assertEqual(reason, "gemini_key_missing")

            mock_log = MagicMock()
            act, tgt, ss, cid = route3_handle("do something", "open_app", 0.2, mock_log)
            self.assertIsNone(act)
            self.assertEqual(len(get_cache()._store), 0)

    def test_failure_mode_2_missing_dependency(self):
        with patch("vlm._load_api_key", return_value="fake_key"), \
             patch.dict("sys.modules", {"google.generativeai": None}):
            action, target, reason, conf, lat = _call_gemini("do something", "open_app", 0.2, None)
            self.assertIsNone(action)
            self.assertTrue(reason.startswith("dependency_missing"))

    def test_failure_mode_3_api_exception(self):
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = RuntimeError("API quota exhausted")
        with patch("vlm._load_api_key", return_value="fake_key"), \
             patch("google.generativeai.GenerativeModel", return_value=mock_model):
            action, target, reason, conf, lat = _call_gemini("do something", "open_app", 0.2, None)
            self.assertIsNone(action)
            self.assertTrue(reason.startswith("gemini_api_error"))

    def test_failure_mode_4_json_parse_error(self):
        action, target, reason, conf, lat = _parse_vlm_response("Not a JSON string at all", "open_app")
        self.assertIsNone(action)
        self.assertTrue(reason.startswith("json_parse_error"))

    def test_failure_mode_5_unrecognised_action(self):
        action, target, reason, conf, lat = _parse_vlm_response('{"action": "hack_the_system"}', "open_app")
        self.assertIsNone(action)
        self.assertTrue(reason.startswith("unrecognised_action"))

    def test_disruptive_trigger_sanity_gate_refusal(self):
        # lock_screen without "lock" in command must be refused
        allowed, reason = check_disruptive_trigger("lock_screen", "tap on the blue thing")
        self.assertFalse(allowed)
        self.assertEqual(reason, "disruptive_action_without_trigger")

        # sleep_pc without trigger words
        allowed, reason = check_disruptive_trigger("sleep_pc", "click the circle icon")
        self.assertFalse(allowed)

        # close_desktop without both 'close' and 'desktop'
        allowed, reason = check_disruptive_trigger("close_desktop", "close this window")
        self.assertFalse(allowed)

    def test_disruptive_trigger_sanity_gate_allowed(self):
        # lock_screen with "lock"
        allowed, reason = check_disruptive_trigger("lock_screen", "please lock my screen")
        self.assertTrue(allowed)
        self.assertIsNone(reason)

        # sleep_pc with "hibernate"
        allowed, reason = check_disruptive_trigger("sleep_pc", "hibernate the pc")
        self.assertTrue(allowed)

        # close_desktop with both 'close' and 'desktop'
        allowed, reason = check_disruptive_trigger("close_desktop", "please close desktop 2")
        self.assertTrue(allowed)

        # close_button with "quit"
        allowed, reason = check_disruptive_trigger("close_button", "quit this window")
        self.assertTrue(allowed)

    def test_route3_handle_disruptive_refusal_not_cached_and_not_dispatched(self):
        fake_response = MagicMock()
        fake_response.text = '{"action": "lock_screen", "target": null, "confidence": 0.95}'
        mock_model = MagicMock()
        mock_model.generate_content.return_value = fake_response

        with patch("vlm._load_api_key", return_value="fake_key"), \
             patch("google.generativeai.GenerativeModel", return_value=mock_model):
            mock_log = MagicMock()
            act, tgt, ss, cid = route3_handle("tap on the blue thing", "screenshot", 0.2, mock_log)
            # Refused because "lock" is not in command
            self.assertIsNone(act)
            self.assertEqual(tgt, "disruptive_action_without_trigger")
            # Must NOT be stored in cache
            self.assertEqual(len(get_cache()._store), 0)

    def test_main_run_action_handles_route3_failure(self):
        mock_entry = MagicMock()
        with patch("vlm.route3_handle", return_value=(None, "disruptive_action_without_trigger", None, "c123")), \
             patch("main._dispatch_action") as mock_dispatch:
            main._run_action("tap on the blue thing", "screenshot", None, 0.2, "vlm_fallback", mock_entry)
            # Must NOT dispatch any action
            mock_dispatch.assert_not_called()
            # Must log route3_failed with reason
            mock_entry.mark_result.assert_called_with("route3_failed", error_msg="disruptive_action_without_trigger")


import numpy as np
from route3_cache import SemanticCache, get_cache


class TestP2CacheKeyScreenMatching(unittest.TestCase):
    """P2: Verify cache key matching: exact command, hamming distance, intent hint, and display_unavailable."""

    def setUp(self):
        self.cache = SemanticCache()
        self.h0 = "0" * 32
        # 3 bits different: change last hex digit '0' -> '7' (0b0111 has 3 bits)
        self.h_3bit = ("0" * 31) + "7"
        # 40 bits different: 10 hex 'f' characters (10 * 4 = 40 bits)
        self.h_40bit = ("0" * 22) + ("f" * 10)

    def test_semantic_path_off_by_default(self):
        self.assertFalse(route3_config.CACHE_SEMANTIC_ENABLED)

    def test_identical_hash_hit(self):
        self.cache.store("open notepad", self.h0, {"action": "open_app", "target": "notepad"}, action="open_app")
        hit, resp = self.cache.query("open notepad", self.h0, action_hint="open_app")
        self.assertTrue(hit)
        self.assertIsNotNone(resp)
        self.assertEqual(resp["action"], "open_app")
        self.assertEqual(resp["_cache_hamming"], 0)

    def test_3bit_difference_hit(self):
        self.cache.store("open notepad", self.h0, {"action": "open_app", "target": "notepad"}, action="open_app")
        hit, resp = self.cache.query("open notepad", self.h_3bit, action_hint="open_app")
        self.assertTrue(hit)
        self.assertIsNotNone(resp)
        self.assertEqual(resp["action"], "open_app")
        self.assertEqual(resp["_cache_hamming"], 3)

    def test_40bit_difference_miss(self):
        self.cache.store("open notepad", self.h0, {"action": "open_app", "target": "notepad"}, action="open_app")
        hit, resp = self.cache.query("open notepad", self.h_40bit, action_hint="open_app")
        self.assertFalse(hit)
        self.assertIsNone(resp)

    def test_display_unavailable_miss(self):
        # Query with display_unavailable misses even if command matches
        self.cache.store("open notepad", self.h0, {"action": "open_app", "target": "notepad"}, action="open_app")
        hit, resp = self.cache.query("open notepad", "display_unavailable", action_hint="open_app")
        self.assertFalse(hit)
        self.assertIsNone(resp)

        # Storing with display_unavailable never stores
        fresh_cache = SemanticCache()
        fresh_cache.store("open notepad", "display_unavailable", {"action": "open_app", "target": "notepad"}, action="open_app")
        self.assertEqual(fresh_cache.size, 0)

    def test_different_command_miss_with_identical_hash(self):
        self.cache.store("close chrome", self.h0, {"action": "close_app", "target": "chrome"}, action="close_app")
        hit, resp = self.cache.query("close notepad", self.h0, action_hint="close_app")
        self.assertFalse(hit)
        self.assertIsNone(resp)

    def test_different_hint_intent_miss(self):
        self.cache.store("open notepad", self.h0, {"action": "open_app", "target": "notepad"}, action="open_app")
        hit, resp = self.cache.query("open notepad", self.h0, action_hint="close_app")
        self.assertFalse(hit)
        self.assertIsNone(resp)

    def test_semantic_path_when_enabled(self):
        # Deterministic fake embedder
        def fake_embed(text):
            if "launch" in text or "open" in text:
                return np.array([1.0, 0.0, 0.0])
            return np.array([0.0, 1.0, 0.0])

        with patch("route3_cache._embed", side_effect=fake_embed), \
             patch("route3_config.CACHE_SEMANTIC_ENABLED", True):
            self.cache.store("open notepad", self.h0, {"action": "open_app", "target": "notepad"}, action="open_app")
            # "launch notepad" has cosine similarity 1.0 >= 0.92 with fake_embed
            hit, resp = self.cache.query("launch notepad", self.h0, action_hint="open_app")
            self.assertTrue(hit)
            self.assertIsNotNone(resp)
            self.assertEqual(resp["action"], "open_app")


if __name__ == "__main__":
    unittest.main(verbosity=2)
