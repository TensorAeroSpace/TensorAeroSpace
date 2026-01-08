import ctypes

from tensoraerospace.aerospacemodel.utils import rtwtypes as rt


def test_scalar_aliases_match_ctypes():
    assert rt.int8_T is ctypes.c_byte
    assert rt.uint8_T is ctypes.c_ubyte
    assert rt.real32_T is ctypes.c_float
    assert rt.real64_T is ctypes.c_double
    assert rt.real_T is ctypes.c_double
    assert rt.time_T is ctypes.c_double


def test_complex_float_struct_layout_and_size():
    z = rt.creal32_T()
    z.re = rt.real32_T(1.5)
    z.im = rt.real32_T(-2.5)
    assert float(z.re) == 1.5
    assert float(z.im) == -2.5
    assert ctypes.sizeof(rt.creal32_T) == ctypes.sizeof(ctypes.c_float) * 2


def test_complex_int_structs_hold_values():
    z64 = rt.cint64_T()
    z64.re = rt.int64_T(10)
    z64.im = rt.int64_T(-3)
    assert z64.re == 10
    assert z64.im == -3

    uz = rt.cuint8_T()
    uz.re = rt.uint8_T(255)
    uz.im = rt.uint8_T(1)
    assert uz.re == 255
    assert uz.im == 1


def test_ext_structs_have_expected_fields_and_defaults():
    y = rt.ExtY_T()
    # ctypes structures are zero-initialized by default
    assert hasattr(y, "u") and hasattr(y, "theta")
    assert float(y.u) == 0.0
    assert float(y.theta) == 0.0

    yr = rt.ExtY_T_r()
    assert hasattr(yr, "w") and hasattr(yr, "time")
    assert float(yr.time) == 0.0
