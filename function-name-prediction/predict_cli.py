import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.inference.predict import predict_function


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python predict_cli.py "your metadata text"')
        sys.exit(1)

    metadata_text = " ".join(sys.argv[1:]).strip()
    if not metadata_text:
        print("Error: metadata input cannot be empty.")
        sys.exit(1)

    try:
        prediction = predict_function(metadata_text)
        print(f"Predicted Function: {prediction}")
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
