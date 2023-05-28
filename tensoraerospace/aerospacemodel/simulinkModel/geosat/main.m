clear;

% init parameters

[A, B, C, D] = geosat_data();

init = [0 0 0.001];
ref_signal = -0.1;

t_s = 0;
t_e = 1500;
dt = 0.1;

sim_out = sim('geosat_model.slx');

y = sim_out.get('yout');

rho = y.getElement(1).Values.Data;
theta = y.getElement(2).Values.Data;
omega = y.getElement(3).Values.Data;
t = y.getElement(4).Values.Data;

bdclose('all');
open('geosat_model.slx');