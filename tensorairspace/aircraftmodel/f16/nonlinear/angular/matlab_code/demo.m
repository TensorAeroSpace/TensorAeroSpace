clear;
clc;
%%
%Временной интервал, шаг дискретизации
t0 = 0; dt = 0.02; tn = 20;
n = (tn - t0) / dt + 1;

%Параметры самолета
parameters = airplane_parameters();

%Начальное состояние
alpha0 = deg2rad(0); beta0 = deg2rad(0);
wx0 = deg2rad(0); wy0 = deg2rad(0); wz0 = deg2rad(0);
gamma0 = deg2rad(0); psi0 = deg2rad(0); theta0 = deg2rad(0);
stab0 = deg2rad(0); ail0 = deg2rad(0); dir0 = deg2rad(0);
dstab0 = deg2rad(0); dail0 = deg2rad(0); ddir0 = deg2rad(0);
x0 = [alpha0; beta0;
      wx0; wy0; wz0;
      gamma0; psi0; theta0;
      stab0; dstab0;
      ail0; dail0;
      dir0; ddir0];
  
%Случайное управление
pd1 = makedist('Uniform', 'lower', -1.0, 'upper', 1.0);
pd2 = makedist('Uniform', 'lower', -0.5, 'upper', 0.5);
pd3 = makedist('Uniform', 'lower', 0.2, 'upper', 1.0);
stab_act = deg2rad(-5 + random_steps(t0, dt, tn, pd1, pd3));
ail_act = deg2rad(random_steps(t0, dt, tn, pd2, pd3));
dir_act = deg2rad(random_steps(t0, dt, tn, pd1, pd3));
u = [stab_act; ail_act; dir_act];
u_0 = [0;0;0];
%%
%Интегрирование методом Эйлера
x = zeros(length(x0), n);
x(:, 1) = x0;
for i = 1 : n-1
    x(:, i+1) = x(:, i) + dt * F16ODE(x(:, i), u_0 , t0 + dt * i, parameters);
end

%%
%Графики
x = F16State_vec2struct(x);
u = F16Control_vec2struct(u);

f = figure;
subplot(3,1,1)
plot(t0:dt:tn, rad2deg(u.stab), '-b'), grid;
ylabel('stab_act, deg');
subplot(3,1,2)
plot(t0:dt:tn, rad2deg(u.ail), '-b'), grid;
ylabel('ail_act, deg');
subplot(3,1,3)
plot(t0:dt:tn, rad2deg(u.dir), '-b'), grid;
ylabel('dir_act, deg');
xlabel('t, sec');

f = figure;
subplot(3,1,1)
plot(t0:dt:tn, rad2deg(x.stab), '-b'), grid;
ylabel('stab, deg');
subplot(3,1,2)
plot(t0:dt:tn, rad2deg(x.ail), '-b'), grid;
ylabel('ail, deg');
subplot(3,1,3)
plot(t0:dt:tn, rad2deg(x.dir), '-b'), grid;
ylabel('dir, deg');
xlabel('t, sec');

f = figure;
subplot(2,1,1)
plot(t0:dt:tn, rad2deg(x.alpha), '-b'), grid;
ylabel('alpha, deg');
subplot(2,1,2)
plot(t0:dt:tn, rad2deg(x.beta), '-b'), grid;
ylabel('beta, deg');
xlabel('t, sec');

f = figure;
subplot(3,1,1)
plot(t0:dt:tn, rad2deg(x.wx), '-b'), grid;
ylabel('wx, deg/sec');
subplot(3,1,2)
plot(t0:dt:tn, rad2deg(x.wy), '-b'), grid;
ylabel('wy, deg/sec');
subplot(3,1,3)
plot(t0:dt:tn, rad2deg(x.wz), '-b'), grid;
ylabel('wz, deg/sec');
xlabel('t, sec');

f = figure;
subplot(3,1,1)
plot(t0:dt:tn, rad2deg(x.gamma), '-b'), grid;
ylabel('gamma, deg');
subplot(3,1,2)
plot(t0:dt:tn, rad2deg(x.psi), '-b'), grid;
ylabel('psi, deg');
subplot(3,1,3)
plot(t0:dt:tn, rad2deg(x.theta), '-b'), grid;
ylabel('theta, deg');
xlabel('t, sec');