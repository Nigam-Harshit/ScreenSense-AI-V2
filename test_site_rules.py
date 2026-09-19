"""
test_site_rules.py  --  Unit tests for deterministic site and scroll rules.

Asserts every row of Appendix Part 1 and Part 2 using standard library unittest.
Mocks webbrowser.open and pyautogui.scroll so tests run with no browser and no screen.
"""

import unittest
from unittest.mock import patch, MagicMock

from site_rules import match_rule, SITE_ALIASES, SCROLL_AMOUNTS, UNSUPPORTED_MESSAGE
import automation
from main import _run_rule_action


class TestSiteRulesMatching(unittest.TestCase):
    """Verify match_rule output for all Appendix Part 1 and Part 2 commands."""

    # ── Appendix Part 1: Must pass (new behavior) ──────────────────────────────
    def test_open_youtube(self):
        self.assertEqual(match_rule("open youtube"), ("open_website", "https://www.youtube.com"))

    def test_click_on_youtube(self):
        self.assertEqual(match_rule("click on youtube"), ("open_website", "https://www.youtube.com"))

    def test_tap_on_youtube(self):
        self.assertEqual(match_rule("tap on youtube"), ("open_website", "https://www.youtube.com"))

    def test_go_to_google(self):
        self.assertEqual(match_rule("go to google"), ("open_website", "https://www.google.com"))

    def test_visit_github(self):
        self.assertEqual(match_rule("visit github"), ("open_website", "https://github.com"))

    def test_launch_gmail(self):
        self.assertEqual(match_rule("launch gmail"), ("open_website", "https://mail.google.com"))

    def test_open_apples_website(self):
        self.assertEqual(match_rule("open apple's website"), ("open_website", "https://www.apple.com"))

    def test_click_on_apples_website(self):
        self.assertEqual(match_rule("click on apple's website"), ("open_website", "https://www.apple.com"))

    def test_click_on_the_apple_website(self):
        self.assertEqual(match_rule("click on the apple website"), ("open_website", "https://www.apple.com"))

    def test_open_the_linkedin_site(self):
        self.assertEqual(match_rule("open the linkedin site"), ("open_website", "https://www.linkedin.com"))

    def test_open_stack_overflow(self):
        self.assertEqual(match_rule("open stack overflow"), ("open_website", "https://stackoverflow.com"))

    def test_open_hugging_face(self):
        self.assertEqual(match_rule("open hugging face"), ("open_website", "https://huggingface.co"))

    def test_scroll_down_a_bit(self):
        self.assertEqual(match_rule("scroll down a bit"), ("scroll", "down", 300))

    def test_scroll_up_a_little(self):
        self.assertEqual(match_rule("scroll up a little"), ("scroll", "up", 300))

    def test_scroll_down_slightly(self):
        self.assertEqual(match_rule("scroll down slightly"), ("scroll", "down", 300))

    def test_scroll_down_a_lot(self):
        self.assertEqual(match_rule("scroll down a lot"), ("scroll", "down", 1600))

    def test_click_on_the_1st_website(self):
        res = match_rule("click on the 1st website")
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "unsupported")

    def test_click_on_the_first_result(self):
        res = match_rule("click on the first result")
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "unsupported")

    def test_tap_on_harshit_nigam(self):
        res = match_rule("tap on harshit nigam")
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "unsupported")

    def test_click_on_second_link(self):
        res = match_rule("click on second link")
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "unsupported")

    def test_open_the_top_result(self):
        res = match_rule("open the top result")
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "unsupported")

    def test_tap_on_top_video(self):
        res = match_rule("tap on top video")
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "unsupported")

    # ── Appendix Part 2: Must fall through (match_rule returns None) ────────────
    def test_part2_open_notepad(self):
        self.assertIsNone(match_rule("open notepad"))

    def test_part2_open_chrome(self):
        self.assertIsNone(match_rule("open chrome"))

    def test_part2_open_google_chrome(self):
        self.assertIsNone(match_rule("open google chrome"))

    def test_part2_open_spotify(self):
        self.assertIsNone(match_rule("open spotify"))

    def test_part2_open_youtube_music(self):
        self.assertIsNone(match_rule("open youtube music"))

    def test_part2_open_settings(self):
        self.assertIsNone(match_rule("open settings"))

    def test_part2_close_youtube(self):
        self.assertIsNone(match_rule("close youtube"))

    def test_part2_scroll_down(self):
        self.assertIsNone(match_rule("scroll down"))

    def test_part2_scroll_up(self):
        self.assertIsNone(match_rule("scroll up"))

    def test_part2_click_close_button_chrome(self):
        self.assertIsNone(match_rule("click on the close button of chrome"))

    def test_part2_minimize_chrome(self):
        self.assertIsNone(match_rule("minimize chrome"))

    def test_part2_move_chrome_left(self):
        self.assertIsNone(match_rule("move chrome left"))

    def test_part2_set_volume_to_30(self):
        self.assertIsNone(match_rule("set volume to 30"))


class TestRuleExecution(unittest.TestCase):
    """Verify execution of matched rules via _run_rule_action with mocked side effects."""

    @patch("webbrowser.open")
    def test_open_website_execution(self, mock_browser_open):
        rule_match = ("open_website", "https://www.youtube.com")
        mock_log = MagicMock()
        _run_rule_action(rule_match, mock_log)
        mock_browser_open.assert_called_once_with("https://www.youtube.com")
        mock_log.set_action_taken.assert_called_once_with("open_website")
        mock_log.mark_result.assert_called_once_with("ok")

    @patch("pyautogui.scroll")
    def test_scroll_down_execution(self, mock_scroll):
        rule_match = ("scroll", "down", 300)
        mock_log = MagicMock()
        _run_rule_action(rule_match, mock_log)
        mock_scroll.assert_called_once_with(-300)
        mock_log.set_action_taken.assert_called_once_with("scroll_down")
        mock_log.mark_result.assert_called_once_with("ok")

    @patch("pyautogui.scroll")
    def test_scroll_up_execution(self, mock_scroll):
        rule_match = ("scroll", "up", 1600)
        mock_log = MagicMock()
        _run_rule_action(rule_match, mock_log)
        mock_scroll.assert_called_once_with(1600)
        mock_log.set_action_taken.assert_called_once_with("scroll_up")
        mock_log.mark_result.assert_called_once_with("ok")

    @patch("pyautogui.scroll")
    def test_scroll_default_amount(self, mock_scroll):
        automation.scroll_down()
        mock_scroll.assert_called_with(-800)
        automation.scroll_up()
        mock_scroll.assert_called_with(800)

    def test_unsupported_execution(self):
        rule_match = ("unsupported", UNSUPPORTED_MESSAGE)
        mock_log = MagicMock()
        _run_rule_action(rule_match, mock_log)
        mock_log.set_action_taken.assert_called_once_with("unsupported")
        mock_log.mark_result.assert_called_once_with("unsupported", error_msg=UNSUPPORTED_MESSAGE)


class TestP5RulePhrasing(unittest.TestCase):
    """P5: Natural speech phrasing rules for site and scroll commands."""

    def test_positive_phrasing_open_youtube_with_please_prefix(self):
        self.assertEqual(match_rule("please open youtube"), ("open_website", "https://www.youtube.com"))

    def test_positive_phrasing_open_apples_website_with_can_you(self):
        self.assertEqual(match_rule("can you open apple's website"), ("open_website", "https://www.apple.com"))

    def test_positive_phrasing_open_up_youtube(self):
        self.assertEqual(match_rule("open up youtube"), ("open_website", "https://www.youtube.com"))

    def test_positive_phrasing_fire_up_github(self):
        self.assertEqual(match_rule("fire up github"), ("open_website", "https://github.com"))

    def test_positive_phrasing_open_youtube_trailing_please(self):
        self.assertEqual(match_rule("open youtube please"), ("open_website", "https://www.youtube.com"))

    def test_positive_phrasing_scroll_down_a_bit_with_could_you(self):
        self.assertEqual(match_rule("could you scroll down a bit"), ("scroll", "down", 300))

    def test_negative_phrasing_open_up_notepad(self):
        self.assertIsNone(match_rule("open up notepad"))

    def test_negative_phrasing_please_open_google_chrome(self):
        self.assertIsNone(match_rule("please open google chrome"))

    def test_negative_phrasing_please_close_youtube(self):
        self.assertIsNone(match_rule("please close youtube"))

    def test_unsupported_guard_with_polite_prefix(self):
        self.assertEqual(
            match_rule("please click on the first result"),
            ("unsupported", UNSUPPORTED_MESSAGE)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)


