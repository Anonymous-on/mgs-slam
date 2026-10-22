"""Make directly vendored upstream packages importable from this source tree."""
from pathlib import Path
import sys


def bootstrap():
    root = Path(__file__).resolve().parent.parent / 'third_party'
    for path in (root, root / 'asmk', root / 'dust3r',
                 root / 'diff-gaussian-rasterization'):
        item = str(path)
        if item not in sys.path:
            sys.path.insert(0, item)
