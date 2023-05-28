clear;
clc;
%%
%Временной интервал, шаг дискретизации
t0 = 0; dt = 0.02; tn = 20;
n = (tn - t0) / dt + 1;

%Параметры самолета
parameters = airplane_parameters();

%Начальное состояние
alpha0 = deg2rad(4);
wz0 = deg2rad(0);
stab0 = deg2rad(-5);
dstab0 = deg2rad(0);
x0 = [alpha0; wz0; stab0; dstab0];
  
%Случайное управление
pd1 = makedist('Uniform', 'lower', -1.0, 'upper', 1.0);
pd2 = makedist('Uniform', 'lower', 0.2, 'upper', 1.0);
stab_act = deg2rad(-5 + random_steps(t0, dt, tn, pd1, pd2));
u = [stab_act];
    
%%
%Интегрирование методом Эйлера
x = zeros(length(x0), n);
x(:, 1) = x0;
for i = 1 : n-1
    x(:, i+1) = x(:, i) + dt * F16ODE(x(:, i), u(:, i), t0 + dt * i, parameters);
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
plot(t0:dt:tn, rad2deg(x.alpha), '-b'), grid;
ylabel('alpha, deg');
subplot(3,1,3)
plot(t0:dt:tn, rad2deg(x.wz), '-b'), grid;
ylabel('wz, deg/sec');
xlabel('t, sec');