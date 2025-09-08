import pytest

from tensoraerospace.agent.base import get_class_from_string


def test_get_class_from_string_import_errors():
    with pytest.raises(ImportError):
        get_class_from_string("no.such.module.Class")
    with pytest.raises(AttributeError):
        get_class_from_string("builtins.NoSuchClass")
