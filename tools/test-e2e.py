#!/usr/bin/env python

from argparse import ArgumentParser, BooleanOptionalAction
import re
import os
from subprocess import run
import sys
import textwrap
import time


def build_ocrs() -> None:
    run("cargo build --release -p ocrs-cli", shell=True, check=True, text=True)


def extract_text(image_path: str) -> str:
    """Extract text from an image using ocrs."""
    result = run(
        # We run the binary directly here rather than use `cargo run` as it
        # is slightly faster.
        [f"target/release/ocrs", image_path],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout


IMAGE_PAT = "\\.(jpeg|jpg|png|webp)$"


def run_tests(test_case_dir: str, *, verbose=False, update_baselines=False) -> bool:
    """
    Compare extracted text for image files against expectations.

    Each image file in `test_case_dir` is expected to have an accompanying
    "{image_name}.expected.txt" file.

    If `update_baselines` is true, mismatches between the actual and expected
    results will result in the expected results being updated. When this flag
    is set, the tests will still succeed if there is a mismatch.

    Returns True if all test cases passed.
    """
    image_filenames = [
        path for path in os.listdir(test_case_dir) if re.search(IMAGE_PAT, path)
    ]

    print(f"Testing {len(image_filenames)} images...")

    errors = 0
    for i, fname in enumerate(image_filenames):
        basename = os.path.splitext(fname)[0]
        expected_path = f"{test_case_dir}/{basename}.expected.txt"
        with open(expected_path) as fp:
            expected_text = fp.read()

        print(f"[{i+1}/{len(image_filenames)}] Testing {fname}", end="")
        start = time.perf_counter()
        text = extract_text(f"{test_case_dir}/{fname}")
        elapsed = time.perf_counter() - start
        print(f" ({elapsed:0.2f}s)")

        expected_text = expected_text.strip()
        text = text.strip()

        if text != expected_text:
            if update_baselines:
                with open(expected_path, 'w') as fp:
                    fp.write(text)
                print(f"Updated baseline for {fname}")
            else:
                errors += 1

                print(f"Actual vs expected mismatch for {fname}")

                if verbose:
                    print("Actual:")
                    print(textwrap.indent(text, "  "))
                    print("Expected:")
                    print(textwrap.indent(expected_text, "  "))

    if errors != 0:
        print(f"{errors} tests failed")

    return errors == 0


parser = ArgumentParser(
    description="""
Run end-to-end tests of ocrs.

Runs ocrs on a set of image files and compares the extracted text with
expectations in `{imagename}.expected.txt` files.
"""
)
parser.add_argument("dir", help="Directory containing test images and expected outputs")
parser.add_argument(
    "-v", "--verbose", action=BooleanOptionalAction, help="Enable verbose logging"
)
parser.add_argument(
    "-u", "--update", action=BooleanOptionalAction, help="Update baselines"
)
args = parser.parse_args()

print("Building ocrs...")
build_ocrs()
passed = run_tests(args.dir, verbose=args.verbose, update_baselines=args.update)

if not passed:
    sys.exit(1)
