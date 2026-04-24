#!/usr/bin/env python3
"""
PRClaw 根目录 CLI 入口。

在 prclaw 项目根目录下可直接执行：
  python3 main.py
  python3 main.py --check

等价于：python3 -m cli.main [参数]
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    from cli.main import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
