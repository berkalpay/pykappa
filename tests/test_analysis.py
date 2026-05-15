import re
import pytest
import shutil

from test_system import heterodimerization_system


@pytest.mark.skipif(not shutil.which("KaSa"), reason="Missing KaSa binary!")
def test_contact_map():
    system = heterodimerization_system()
    pattern = re.compile(r"\s+")

    # strip whitespace
    src = re.sub(pattern, "", system.contact_map().unflatten().source)

    reference = """
        graph G {
                subgraph cluster0 {
                        graph [label=A,
                                shape=box
                        ];
                        0.0     [label=x,
                                shape=circle,
                                size=5];
                }
                subgraph cluster1 {
                        graph [label=B,
                                shape=box
                        ];
                        1.0     [label=x,
                                shape=circle,
                                size=5];
                }
                0.0 -- 1.0;
        }
    """

    assert src == re.sub(pattern, "", reference)
