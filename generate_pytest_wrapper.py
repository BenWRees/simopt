#!/usr/bin/env python3

"""Generate pytest wrapper utilities."""

import os
import sys
import textwrap

OUTPUT_FILE = sys.argv[1]
FILES = sys.argv[2:]


def module_name(path: str) -> str:
    """Module name."""
    return path[:-3].replace(os.sep, ".")


with open(OUTPUT_FILE, "w") as f:  # noqa: PTH123
    f.write("import pytest\n\n")

    for i, file_path in enumerate(FILES):
        mod = module_name(file_path)

        test_code = f"""
def test_file_{i}():
    try:
        import {mod} as module
        if hasattr(module, "main"):
            module.main()
    except Exception as e:
        pytest.fail(f"Runtime error in {file_path}: {{e}}")
"""
        f.write(textwrap.dedent(test_code))
