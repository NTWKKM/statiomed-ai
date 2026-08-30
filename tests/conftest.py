import os
import subprocess
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Forwarding wrapper if invoked via system Python (e.g. Python 3.9) or outside .venv
VENV_PYTEST = REPO_ROOT / ".venv" / "bin" / "pytest"
if (
    sys.version_info < (3, 10)
    and VENV_PYTEST.exists()
    and not os.environ.get("_PYTEST_FORWARDED")
):
    os.environ["_PYTEST_FORWARDED"] = "1"
    args = [str(VENV_PYTEST)] + [a for a in sys.argv[1:] if a != "pytest"]
    res = subprocess.run(args)
    sys.exit(res.returncode)
