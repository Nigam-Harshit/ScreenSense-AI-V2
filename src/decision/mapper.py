def map_action(action, target):
    if action == "move_mouse":
        return {"type": "move", "direction": target}

    elif action == "click":
        return {"type": "click"}

    elif action == "close_notepad":
        return {"type": "vision_click", "template": "data/close_button.png"}

    return None
