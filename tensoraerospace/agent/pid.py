import numpy as np
from tensoraerospace.benchmark.function import overshoot, settling_time, static_error
import torch 

class PIDController:
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = 0
        self.prev_error = 0

    def update(self, setpoint, measurement):
        error = setpoint - measurement
        self.integral = self.integral + error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.prev_error = error
        return output