def parse_command(command):
    command = command.lower()

    if "move right" in command:
        return "move_mouse", "right"
    elif "move left" in command:
        return "move_mouse", "left"
    elif "move up" in command:
        return "move_mouse", "up"
    elif "move down" in command:
        return "move_mouse", "down"
    elif "click" in command:
        return "click", None
    elif "close notepad" in command:
        return "close_notepad", None
    else:
        return None, None
    
    
