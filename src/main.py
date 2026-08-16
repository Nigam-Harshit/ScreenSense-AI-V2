import ctypes
ctypes.windll.user32.SetProcessDPIAware()

from src.nlp.parser import parse_command
from src.decision.mapper import map_action
from src.automation.executor import execute
from src.vision.detector import find_template

def main():
    print("ScreenSense AI Running")

    while True:
        command = input("Enter command (exit to stop): ")

        if command.lower() == "exit":
            break

        action, target = parse_command(command)
        plan = map_action(action, target)

        print("Action:", action)
        print("Plan:", plan)

        if not plan:
            print("Command not understood")
            continue

        if plan["type"] == "vision_click":
            print("Searching for template...")
            coords = find_template(plan["template"])
            print("Coordinates returned:", coords)

            if coords:
                execute({"type": "click_at", "coords": coords})
            else:
                print("Template not found")

        else:
            execute(plan)

if __name__ == "__main__":
    main()
