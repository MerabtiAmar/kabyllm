"""UTF-8 console setup.

Windows terminals default to cp1252, which cannot encode ɣ (U+0263) or ɛ (U+025B).
Printing a Kabyle word then raises UnicodeEncodeError and kills the script -- a
particularly unhelpful failure mode for a Kabyle NLP toolkit. Every entry point calls
`setup()` before writing anything.
"""

from __future__ import annotations

import logging
import sys


def setup(level: int = logging.INFO, fmt: str = "%(message)s") -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
    logging.basicConfig(level=level, format=fmt)
