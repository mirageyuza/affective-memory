"""包级命令行入口：``python -m emotion_memory`` 打印永恒誓言卡。

封印完好返回码 0；誓词原文被改动（封印失效）返回码 1，
便于 CI 或发布流程把"誓言被篡改"当作硬失败拦截。
"""
from __future__ import annotations

from .creed import creed_card, seal_intact

if __name__ == "__main__":  # pragma: no cover
    _intact = seal_intact()
    print(creed_card(lapsed=not _intact))
    raise SystemExit(0 if _intact else 1)
