from pathlib import Path
import sys
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('reference_dir', type=str)
    parser.add_argument('result_dir', type=str)
    args = parser.parse_args()

    reference_dir = Path(args.reference_dir)
    result_dir = Path(args.result_dir)
    error_found = False

    for reference_file in reference_dir.rglob('*'):
        if reference_file.is_file():
            relative_path = reference_file.relative_to(reference_dir)
            result_file = result_dir / relative_path

            if not result_file.exists():
                error_found = True
                print(f"File {relative_path} not found in result directory")

    if error_found:
        sys.exit(1)
    else:
        sys.exit(0)
