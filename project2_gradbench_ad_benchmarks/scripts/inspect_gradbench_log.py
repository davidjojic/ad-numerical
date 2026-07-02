import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=Path)
    args = parser.parse_args()

    messages = {}
    evaluation_count = 0
    failed_count = 0
    invalid_count = 0

    with args.log_file.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            message = record.get("message")
            if isinstance(message, dict):
                message_id = message.get("id")

                if message_id is not None:
                    messages[message_id] = message

                if message.get("kind") == "analysis" and message.get("valid") is False:
                    invalid_count += 1

            response = record.get("response")
            if not isinstance(response, dict):
                continue

            if response.get("success") is False:
                failed_count += 1

            response_id = response.get("id")
            original_message = messages.get(response_id)

            if not original_message:
                continue

            if original_message.get("kind") != "evaluate":
                continue

            evaluation_count += 1

            module = original_message.get("module")
            function = original_message.get("function")
            timings = response.get("timings", [])

            print(f"Evaluation {evaluation_count}")
            print(f"  Module: {module}")
            print(f"  Function: {function}")
            print(f"  Success: {response.get('success')}")
            print(f"  Timings: {timings}")
            print()

    print("Summary")
    print(f"  Evaluations: {evaluation_count}")
    print(f"  Failed responses: {failed_count}")
    print(f"  Invalid analyses: {invalid_count}")


if __name__ == "__main__":
    main()