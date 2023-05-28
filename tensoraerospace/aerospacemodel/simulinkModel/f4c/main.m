clear;

% init parameters

[A, B, C, D] = f4c_data();

init = [1452 0.0 0 0];
ref_signal = -0.2;

t_s = 0;
t_e = 500;
dt = 0.1;

sim_out = sim('f4c_model.slx');

y = sim_out.get('yout');

u = y.getElement(1).Values.Data;
w = y.getElement(2).Values.Data;
q = y.getElement(3).Values.Data;
theta = y.getElement(4).Values.Data;
t = y.getElement(5).Values.Data;

bdclose('all');
open('f4c_model.slx');