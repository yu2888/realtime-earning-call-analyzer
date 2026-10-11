import json
from pathlib import Path
import sys
import time


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "Usage: stream_terminal_viewer.py <event-file> <ready-file> <done-file>"
        )

    event_path = Path(sys.argv[1])
    ready_path = Path(sys.argv[2])
    done_path = Path(sys.argv[3])
    print("LLM stream (right terminal)", flush=True)
    received_stop = False
    try:
        with event_path.open(encoding="utf-8") as event_file:
            ready_path.touch()
            while True:
                line = event_file.readline()
                if not line:
                    time.sleep(0.05)
                    continue

                event = json.loads(line)
                event_type = event["type"]
                text = event["text"]
                if event_type == "context":
                    print(f"\n[LLM input]\n{text}\n[LLM response]", flush=True)
                elif event_type == "token":
                    print(text, end="", flush=True)
                elif event_type == "response_end":
                    print(flush=True)
                elif event_type == "status":
                    print(f"\n[{text}]", flush=True)
                elif event_type == "stop":
                    print("\nLLM stream ended.", flush=True)
                    received_stop = True
                    return
                else:
                    raise RuntimeError(f"Unknown LLM event type: {event_type}")
    finally:
        if received_stop:
            done_path.touch()


if __name__ == "__main__":
    main()
