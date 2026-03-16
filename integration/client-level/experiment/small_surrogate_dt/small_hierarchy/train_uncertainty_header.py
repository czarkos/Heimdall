#!/usr/bin/env python3

import os
import runpy
import sys


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(script_dir, "../../hierarchy/train_uncertainty_header.py")
    sys.argv[0] = target
    runpy.run_path(target, run_name="__main__")

