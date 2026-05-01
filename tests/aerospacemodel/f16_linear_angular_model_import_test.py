import importlib


def test_angular_model_import_and_attrs(tmp_path):
    angular_pkg = importlib.import_module(
        "tensoraerospace.aerospacemodel.f16.linear.angular"
    )
    assert hasattr(angular_pkg, "initial_state")
    assert hasattr(angular_pkg, "set_initial_state")

    angular_model = importlib.import_module(
        "tensoraerospace.aerospacemodel.f16.linear.angular.model"
    )
    # smoke: just check class is available
    assert hasattr(angular_model, "AngularF16")
