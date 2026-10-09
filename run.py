#!/usr/bin/env python3
"""Run OpenVat straight from the source tree, without installing it:

    python run.py [model.stl ...]

Only the dependencies in requirements.txt are needed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openvat.app import main  # noqa: E402

sys.exit(main())
