clear;

% init parameters

[A, B, C, D] = x15_data();

init = [1936 0.0 0 0];
ref_signal = -0.2;

t_s = 0;
t_e = 500;
dt = 0.1;

sim_out = sim('x15_model.slx');

y = sim_out.get('yout');

u = y.getElement(1).Values.Data;
w = y.getElement(2).Values.Data;
q = y.getElement(3).Values.Data;
theta = y.getElement(4).Values.Data;
t = y.getElement(5).Values.Data;

bdclose('all');
open('x15_model.slx');