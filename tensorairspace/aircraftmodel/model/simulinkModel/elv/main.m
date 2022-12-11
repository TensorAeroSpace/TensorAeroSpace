clear;

% init parameters

[A, B, C, D] = elv_data();

init = [0 0 0.0];
ref_signal = -0.01;

t_s = 0;
t_e = 500;
dt = 0.1;

sim_out = sim('elv_model.slx');

y = sim_out.get('yout');

w = y.getElement(1).Values.Data;
q = y.getElement(2).Values.Data;
theta = y.getElement(3).Values.Data;
t = y.getElement(4).Values.Data;

bdclose('all');
open('elv_model.slx');