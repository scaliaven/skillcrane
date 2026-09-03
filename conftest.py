"""Put the repo root on sys.path so tests import the flat modules directly."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
