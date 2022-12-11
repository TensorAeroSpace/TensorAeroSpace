clear;

% init parameters

[A, B, C, D] = f16_data();

init = [160 0.628 0 0];
ref_signal = -0.191;

t_s = 0;
t_e = 500;
dt = 0.1;

sim_out = sim('f16_model.slx');

y = sim_out.get('yout');

u = y.getElement(1).Values.Data;
alpha = y.getElement(2).Values.Data;
q = y.getElement(3).Values.Data;
theta = y.getElement(4).Values.Data;
t = y.getElement(5).Values.Data;

bdclose('all');
open('f16_model.slx');