import random
import re
import pytest
import shutil
from pathlib import Path

from pykappa.system import System

from test_system import heterodimerization_system


@pytest.mark.skipif(not shutil.which("KaSa"), reason="Missing KaSa binary!")
def test_contact_map():
    system = heterodimerization_system()
    pattern = re.compile(r"\s+")

    # strip whitespace
    src = re.sub(pattern, "", system.contact_map().unflatten().source)

    with open(str(Path(__file__).parent / "hdimer_cm.dot")) as f:
        cg = re.sub(pattern, "", f.read())
        assert cg == src
