import subprocess
from typing import Sequence

import structlog

logger = structlog.get_logger(__name__)


def run_complexity(extra_args: Sequence[str] | None = None) -> int:
    """Run radon cyclomatic complexity analysis."""
    command = ["radon", "cc", ".", "-nc"]
    if extra_args:
        command.extend(extra_args)
    logger.info("running complexity analysis", command=" ".join(command))
    result = subprocess.run(command)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(run_complexity())
