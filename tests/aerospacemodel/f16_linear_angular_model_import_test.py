import importlib
import sys
import types


def test_angular_model_import_and_attrs(tmp_path):
    # install fake matlab before import
    fake = types.ModuleType("matlab")
    fake.double = lambda x: x
    sys.modules["matlab"] = fake

    angular_model = importlib.import_module(
        "tensoraerospace.aerospacemodel.f16.linear.angular.model"
    )
    # smoke: just check class is available
    assert hasattr(angular_model, "AngularF16")
