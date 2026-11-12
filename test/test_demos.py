import os
import subprocess
import sys
from pathlib import Path

import pytest

demo_files = sorted((Path(__file__).parent / ".." / "demo").glob("*.py"))


@pytest.mark.parametrize("demo", demo_files, ids=lambda demo: demo.name)
def test_demo(demo: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-P", demo],
        env={**os.environ, "CI": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout
