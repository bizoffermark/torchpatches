import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torchpatches


def test_import_version():
    print(torchpatches.__version__)
    assert True
