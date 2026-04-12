#!/usr/bin/env python3
"""Root validator entrypoint."""

from __future__ import annotations

import sys

from app.env.runner import main


if __name__ == "__main__":
    sys.exit(main())
