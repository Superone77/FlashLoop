"""Build the optional CUDA reader against the installed PyTorch and CUDA toolkit."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    package_dir = Path(__file__).resolve().parent
    setup_py = package_dir / "csrc" / "setup.py"
    if not setup_py.is_file():
        raise RuntimeError("FlashLoop CUDA sources are missing from this installation")
    with tempfile.TemporaryDirectory(prefix="flashloop-build-") as build_dir:
        subprocess.run(
            [
                sys.executable,
                str(setup_py),
                "build_ext",
                "--build-lib",
                str(package_dir.parent),
                "--build-temp",
                build_dir,
            ],
            cwd=build_dir,
            check=True,
        )


if __name__ == "__main__":
    main()
