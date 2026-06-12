import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from test_system import heterodimerization_system

if __name__ == "__main__":
    system = heterodimerization_system()
    while system.time < 2:
        system.update()
