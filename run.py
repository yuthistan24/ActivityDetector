#!/usr/bin/env python3
"""ActivityDetector launcher script.

Run GUI mode:
    python run.py

Run preflight system health check:
    python run.py --preflight

Run headless mode:
    python run.py --headless

Custom procedure:
    python run.py --procedure procedures/sample_circuit_assembly.yaml
"""

import sys
from activity_detector.cli import main

if __name__ == "__main__":
    sys.exit(main())
