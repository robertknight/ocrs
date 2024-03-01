import argparse
import subprocess
import sys


def find_images_with_text(files: list[str], text: str) -> list[str]:
    """
    Search for images in `files` containing the given text.

    Returns a list of matching file paths.
    """
    allowed_extensions = (".png", ".jpg", ".jpeg", ".webp")
    paths = []
    files = [fname for fname in files if fname.lower().endswith(allowed_extensions)]

    for i, filename in enumerate(files):
        print(f"\rReading image {i+1}/{len(files)}...", end="", file=sys.stderr)

        try:
            result = subprocess.run(
                ["ocrs", filename], capture_output=True, text=True, check=True
            )
            if text.lower() in result.stdout.lower():
                paths.append(filename)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {filename}: {e}", file=sys.stderr)

    # Add new line after progress message
    print("", file=sys.stderr)

    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for text in images")
    parser.add_argument(
        "files", help="Files to search. Non-image files are ignored.", nargs="*"
    )
    parser.add_argument("text", help="Text to search for")

    args = parser.parse_args()

    paths = find_images_with_text(args.files, args.text)
    for path in paths:
        print(path)
