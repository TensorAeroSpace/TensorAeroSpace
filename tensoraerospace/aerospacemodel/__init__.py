"""
Модуль аэрокосмических моделей TensorAeroSpace.

Этот модуль содержит коллекцию математических моделей различных летательных аппаратов
и космических объектов, включая самолеты, ракеты, спутники и БПЛА. Модели предназначены
для использования в задачах управления, моделирования и обучения с подкреплением.

Доступные модели:
- LongitudinalB747: Модель продольного движения Boeing 747
- ComSat: Модель спутника связи
- ELVRocket: Модель ракеты-носителя
- LongitudinalF4C: Модель продольного движения F-4C Phantom II
- LongitudinalF16: Модель продольного движения F-16 Fighting Falcon
- GeoSat: Модель геостационарного спутника
- LAPAN: Модель самолета наблюдения LAPAN LSU-05 NG
- MissileModel: Модель ракеты
- LongitudinalUAV: Модель продольного движения БПЛА
- Ultrastick: Модель самолета Ultrastick-25e
- LongitudinalX15: Модель продольного движения X-15
"""

from .b747 import LongitudinalB747 as LongitudinalB747
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
