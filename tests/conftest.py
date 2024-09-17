import pytest


@pytest.fixture()
def classes():
    # class_type = "cyst"
    # class_type = "normal"
    # class_type = "stone"
    class_type = "tumor"

    return class_type
