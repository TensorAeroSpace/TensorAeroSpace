import ctypes
import os

import matplotlib.pyplot as plt
from rtwtypes import *


class ExtY(ctypes.Structure):
    """
    Output parameters Simulink model
    (name, type)
    """

    _fields_ = [
        ("u", real_T),
        ("w", real_T),
        ("q", real_T),
        ("theta", real_T),
        ("sim_time", real_T),
    ]


class ExtU(ctypes.Structure):
    """
    INput parameters Simulink model
    (name, type)
    """

    _fields_ = [
        ("ref_signal", real_T),
    ]


dll_path = os.path.abspath("model.dll")
dll = ctypes.cdll.LoadLibrary(dll_path)

X = ExtU.in_dll(dll, "uav1_model_U")
Y = ExtY.in_dll(dll, "uav1_model_Y")

model_initialize = dll.model_initialize
model_step = dll.model_step
model_terminate = dll.model_terminate


model_initialize()

u = []
w = []
q = []
theta = []

for step in range(int(2100)):
    X.ref_signal = -0.0
    model_step()
    u.append(Y.u)
    w.append(Y.w)
    q.append(Y.q)
    theta.append(Y.theta)

model_terminate()


plt.plot(wz)

plt.ylabel("$u$, [м/с]")

plt.savefig("u.png")
