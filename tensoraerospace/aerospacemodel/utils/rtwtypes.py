"""
Модуль определений типов данных для интеграции с Simulink/MATLAB.

Этот модуль содержит определения типов данных, совместимых с Real-Time Workshop (RTW)
и Simulink Coder, используемых для генерации C-кода из моделей Simulink.
Включает базовые типы данных фиксированной ширины, комплексные числа и структуры
для обмена данными с моделями Simulink.
"""

import ctypes

"""
  Fixed width word size data types:                                     
    int8_T, int16_T, int32_T     - signed 8, 16, or 32 bit integers     
    uint8_T, uint16_T, uint32_T  - unsigned 8, 16, or 32 bit integers   
    real32_T, real64_T           - 32 and 64 bit floating point numbers 

"""
int8_T = ctypes.c_byte
uint8_T = ctypes.c_ubyte
int16_T = ctypes.c_short
uint16_T = ctypes.c_ushort
int32_T = ctypes.c_int
uint32_T = ctypes.c_uint
int64_T = ctypes.c_longlong
uint64_T = ctypes.c_ulonglong
real32_T = ctypes.c_float
real64_T = ctypes.c_double
"""

  Generic type definitions: boolean_T, char_T, byte_T, int_T, uint_T,       
                            real_T, time_T, ulong_T, ulonglong_T.           
"""
real_T = ctypes.c_double
time_T = ctypes.c_double
boolean_T = ctypes.c_ubyte
int_T = ctypes.c_int
uint_T = ctypes.c_uint
ulong_T = ctypes.c_ulong
ulonglong_T = ctypes.c_ulonglong
char_T = ctypes.c_char
uchar_T = ctypes.c_ubyte
char_T = ctypes.c_byte

"""
    Complex number:
        creal32_T, creal64_T, creal_T - 32, 64, 16 bit 
        cint8_T, cint16_T, cint32_T, cint64_T - 8, 16, 32, 64 bit 
        cuint8_T, cuint16_T, cuint32_T, cuint64_T - 8, 16, 32, 64 bit 
"""


class creal32_T(ctypes.Structure):
    """
    Комплексное число с 32-битными компонентами

    Структура для представления комплексного числа с действительной
    и мнимой частями типа real32_T (32-битное число с плавающей точкой).
    """

    _fields_ = [
        ("re", real32_T),
        ("im", real32_T),
    ]


class creal64_T(ctypes.Structure):
    """
    Комплексное число с 64-битными компонентами

    Структура для представления комплексного числа с действительной
    и мнимой частями типа real64_T (64-битное число с плавающей точкой).
    """

    _fields_ = [
        ("re", real64_T),
        ("im", real64_T),
    ]


class creal_T(ctypes.Structure):
    """
    Комплексное число с компонентами типа real_T

    Структура для представления комплексного числа с действительной
    и мнимой частями типа real_T (double precision).
    """

    _fields_ = [
        ("re", real_T),
        ("im", real_T),
    ]


class cint8_T(ctypes.Structure):
    """
    Комплексное число с 8-битными знаковыми целыми компонентами

    Структура для представления комплексного числа с действительной
    и мнимой частями типа int8_T (8-битное знаковое целое).
    """

    _fields_ = [
        ("re", int8_T),
        ("im", int8_T),
    ]


class cuint8_T(ctypes.Structure):
    """
    Комплексное число с 8-битными беззнаковыми целыми компонентами

    Структура для представления комплексного числа с действительной
    и мнимой частями типа uint8_T (8-битное беззнаковое целое).
    """

    _fields_ = [
        ("re", uint8_T),
        ("im", uint8_T),
    ]


class cint16_T(ctypes.Structure):
    """
    Комплексное число с 16-битными знаковыми целыми компонентами

    Структура для представления комплексного числа с действительной
    и мнимой частями типа int16_T (16-битное знаковое целое).
    """

    _fields_ = [
        ("re", int16_T),
        ("im", int16_T),
    ]


class cuint16_T(ctypes.Structure):
    """
    Комплексное число с 16-битными беззнаковыми целыми компонентами

    Структура для представления комплексного числа с действительной
    и мнимой частями типа uint16_T (16-битное беззнаковое целое).
    """

    _fields_ = [
        ("re", uint16_T),
        ("im", uint16_T),
    ]


class cint32_T(ctypes.Structure):
    """
    Комплексное число с 32-битными знаковыми целыми компонентами

    Структура для представления комплексного числа с действительной
    и мнимой частями типа int32_T (32-битное знаковое целое).
    """

    _fields_ = [
        ("re", int32_T),
        ("im", int32_T),
    ]


class cuint32_T(ctypes.Structure):
    """
    Комплексное число с 32-битными беззнаковыми целыми компонентами

    Структура для представления комплексного числа с действительной
    и мнимой частями типа uint32_T (32-битное беззнаковое целое).
    """

    _fields_ = [
        ("re", uint32_T),
        ("im", uint32_T),
    ]


class cint64_T(ctypes.Structure):
    """
    Комплексное число с 64-битными знаковыми целыми компонентами

    Структура для представления комплексного числа с действительной
    и мнимой частями типа int64_T (64-битное знаковое целое).
    """

    _fields_ = [
        ("re", int64_T),
        ("im", int64_T),
    ]


class cuint64_T(ctypes.Structure):
    """
    Комплексное число с 64-битными беззнаковыми целыми компонентами

    Структура для представления комплексного числа с действительной
    и мнимой частями типа uint64_T (64-битное беззнаковое целое).
    """

    _fields_ = [
        ("re", uint64_T),
        ("im", uint64_T),
    ]


class ExtY_T(ctypes.Structure):
    """
    Output parameters Simulink model
    (name, type)
    """

    _fields_ = [
        ("u", real_T),
        ("w", real_T),
        ("q", real_T),
        ("theta", real_T),
        ("time", real_T),
    ]


class ExtY_T_r(ctypes.Structure):
    """
    Output parameters Simulink model short
    (name, type)
    """

    _fields_ = [
        ("w", real_T),
        ("q", real_T),
        ("theta", real_T),
        ("time", real_T),
    ]
