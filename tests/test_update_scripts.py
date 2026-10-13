"""Exercise updater failure reporting without git, network, or package installs."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("failure", [None, "git", "pip-upgrade", "dependencies", "missing-venv"])
def test_linux_update_reports_only_complete_success(tmp_path, failure):
    root = Path(__file__).resolve().parents[1]
    shutil.copy(root / "update.sh", tmp_path / "update.sh")
    (tmp_path / ".git").mkdir()
    commands = tmp_path / "commands"
    commands.mkdir()
    git = commands / "git"
    git.write_text('#!/bin/bash\necho git >> "$UPDATE_TEST_LOG"\n[ "$UPDATE_TEST_FAILURE" != git ]\n')
    git.chmod(0o755)
    if failure != "missing-venv":
        python = tmp_path / "venv/bin/python"
        python.parent.mkdir(parents=True)
        python.write_text('''#!/bin/bash
if [ "$4" = "--upgrade" ]; then
    stage=pip-upgrade
else
    stage=dependencies
fi
echo "$stage" >> "$UPDATE_TEST_LOG"
[ "$UPDATE_TEST_FAILURE" != "$stage" ]
''')
        python.chmod(0o755)
    log = tmp_path / "calls"
    env = dict(os.environ, PATH=f"{commands}:{os.environ['PATH']}",
               UPDATE_TEST_FAILURE=failure or "", UPDATE_TEST_LOG=str(log))
    result = subprocess.run(["bash", str(tmp_path / "update.sh")],
                            env=env, capture_output=True, text=True)
    assert (result.returncode == 0) == (failure is None), result.stdout
    assert ("[SUCCESS]" in result.stdout) == (failure is None)
    expected = {
        None: ["git", "pip-upgrade", "dependencies"],
        "git": ["git"],
        "pip-upgrade": ["git", "pip-upgrade"],
        "dependencies": ["git", "pip-upgrade", "dependencies"],
        "missing-venv": [],
    }
    assert (log.read_text().splitlines() if log.exists() else []) == expected[failure]
