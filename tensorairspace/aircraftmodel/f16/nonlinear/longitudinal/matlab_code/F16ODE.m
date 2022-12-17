function [ Dx ] = F16ODE( x, u, t, p )
% --------------------------------------------------------------
%      Уравнения продольного углового движения F-16.
% --------------------------------------------------------------
% Dx=F16ODE_traj(x,u,t,p) вычисляет правую часть системы ОДУ, описывающих движение F-16 в
%    зависимости от вектора состояния, управления, времени и параметров.
% x - вектор состояния.
% u - вектор управления.
% t - время.
% p - параметры самолета.
% x = [alpha wz stab dstab]^T
% u = [stab_act]^T
% --------------------------------------------------------------
x = F16State_vec2struct(x);
u = F16Control_vec2struct(u);

%Коэффициенты аэродинамических сил
cy = GetCy(x.alpha, deg2rad(0), x.stab, p.lef, x.wz, p.V, p.bA, p.sb);

%Коэффициенты аэродинамических моментов
mz = GetMz(x.alpha, deg2rad(0), x.stab, p.lef, x.wz, p.V, p.bA, p.sb);

%Аэродинамические силы
Y = p.q.*p.S.*cy;

%Аэродинамические моменты
Mz = p.q.*p.S.*p.bA.*mz;

%Результирующие силы и моменты
Ry = Y;
MRz = Mz + p.rcgx.*Ry;

Dx = x;

%Уравнения моментов
Dx.wz = MRz./p.Jz;

%Уравнения сил
gay = -p.g;
Dx.alpha = x.wz - (Ry + p.m.*gay)./(p.m.*p.V);

%Модель приводов, с ограничениями
Dx.stab = min(max(x.dstab, -p.maxabsdstab), p.maxabsdstab);
control_stab = min(max(u.stab, -p.maxabsstab), p.maxabsstab);
Dx.dstab = (-2.*p.Tstab.*p.Xistab.*x.dstab - x.stab + control_stab)./(p.Tstab.^2);

Dx = cell2mat(struct2cell(Dx));

end

