"""Модуль аэрокосмических моделей TensorAeroSpace.

Содержит коллекцию математических моделей различных ЛА и космических объектов
для задач управления, моделирования и обучения с подкреплением.
"""

from .b747 import LongitudinalB747 as LongitudinalB747
from .cessna170 import LongitudinalCessna170 as LongitudinalCessna170
from .comsat import ComSat as ComSat
from .elv import ELVRocket as ELVRocket
from .f4c import LongitudinalF4C as LongitudinalF4C
from .f16.linear.longitudinal.model import LongitudinalF16 as LongitudinalF16
from .geosat import GeoSat as GeoSat
from .lapan import LAPAN as LAPAN
from .rocket import MissileModel as MissileModel
from .uav import LongitudinalUAV as LongitudinalUAV
from .ultrastick import Ultrastick as Ultrastick
from .x15 import LongitudinalX15 as LongitudinalX15

__all__ = [
    "LongitudinalB747",
    "ComSat",
    "ELVRocket",
    "LongitudinalF4C",
    "LongitudinalF16",
    "GeoSat",
    "LAPAN",
    "MissileModel",
    "LongitudinalUAV",
    "Ultrastick",
    "LongitudinalX15",
    "LongitudinalCessna170",
]
