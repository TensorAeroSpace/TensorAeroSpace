function [ Dx ] = F16ODE( x, u, t, p )
% --------------------------------------------------------------
%      Уравнения углового движения F-16.
% --------------------------------------------------------------
% Dx=F16ODE_traj(x,u,t,p) вычисляет правую часть системы ОДУ, описывающих движение F-16 в
%    зависимости от вектора состояния, управления, времени и параметров.
% x - вектор состояния.
% u - вектор управления.
% t - время.
% p - параметры самолета.
% x = [alpha beta
%     wx wy wz
%     gamma psi theta
%     stab dstab
%     ail dail
%     dir ddir]^T
% u = [stab_act ail_act dir_act]^T
% --------------------------------------------------------------
x = F16State_vec2struct(x);
u = F16Control_vec2struct(u);

%Коэффициенты аэродинамических сил
cx = GetCx(x.alpha, x.beta, x.stab, p.lef, x.wz, p.V, p.bA, p.sb);
cy = GetCy(x.alpha, x.beta, x.stab, p.lef, x.wz, p.V, p.bA, p.sb);
cz = GetCz(x.alpha, x.beta, x.dir, x.ail, p.lef, x.wx, x.wy, p.V, p.l);

%Коэффициенты аэродинамических моментов
mx = GetMx(x.alpha, x.beta, x.stab, x.dir, x.ail, p.lef, x.wx, x.wy, p.V, p.l);
my = GetMy(x.alpha, x.beta, x.stab, x.dir, x.ail, p.lef, x.wx, x.wy, p.V, p.l);
mz = GetMz(x.alpha, x.beta, x.stab, p.lef, x.wz, p.V, p.bA, p.sb);

%Аэродинамические силы
X = -p.q.*p.S.*cx;
Y = p.q.*p.S.*cy;
Z = p.q.*p.S.*cz;

%Аэродинамические моменты
Mx = p.q.*p.S.*p.l.*mx;
My = p.q.*p.S.*p.l.*my;
Mz = p.q.*p.S.*p.bA.*mz;

%Результирующие силы и моменты
Rx = X;
Ry = Y;
Rz = Z;
MRx = Mx;
MRy = My - p.rcgx.*Rz;
MRz = Mz + p.rcgx.*Ry;

Dx = x;

%Уравнения моментов
Gamma = p.Jx.*p.Jy - p.Jxy.^2;
Dx.wx = (p.Jy.*MRx + p.Jxy.*(MRy - p.hEx.*x.wz) + p.Jxy.*(p.Jz - p.Jx - p.Jy).*x.wx.*x.wz + (p.Jxy.^2 + p.Jy.*(p.Jy - p.Jz)).*x.wy.*x.wz)./Gamma;
Dx.wy = (p.Jxy.*MRx + p.Jx.*(MRy - p.hEx.*x.wz) + (p.Jx.*(p.Jz - p.Jx) - p.Jxy.^2).*x.wx.*x.wz + p.Jxy.*(p.Jx + p.Jy - p.Jz).*x.wy.*x.wz)./Gamma;
Dx.wz = (MRz + p.hEx.*x.wy + p.Jxy.*(x.wx.^2 - x.wy.^2) + (p.Jx - p.Jy).*x.wx.*x.wy)./p.Jz;

%Уравнения сил
gax = p.g.*(-sin(x.theta).*cos(x.alpha).*cos(x.beta) + cos(x.gamma).*cos(x.theta).*sin(x.alpha).*cos(x.beta) + sin(x.gamma).*cos(x.theta).*sin(x.beta));
gay = p.g.*(-sin(x.theta).*sin(x.alpha) - cos(x.gamma).*cos(x.theta).*cos(x.alpha));
gaz = p.g.*(sin(x.theta).*cos(x.alpha).*sin(x.beta) - cos(x.gamma).*cos(x.theta).*sin(x.alpha).*sin(x.beta) + sin(x.gamma).*cos(x.theta).*cos(x.beta));
Xa = -cos(x.alpha).*cos(x.beta).*Rx - sin(x.alpha).*cos(x.beta).*Ry + sin(x.beta).*Rz;
Ya = -sin(x.alpha).*Rx + cos(x.alpha).*Ry;
Za = cos(x.alpha).*sin(x.beta).*Rx + sin(x.alpha).*sin(x.beta).*Ry + cos(x.beta).*Rz;
Dx.alpha = x.wz + (x.wy.*sin(x.alpha) - x.wx.*cos(x.alpha)).*tan(x.beta) - (Ya + p.m.*gay)./(p.m.*p.V.*cos(x.beta));
Dx.beta = x.wx.*sin(x.alpha) + x.wy.*cos(x.alpha) + (Za + p.m.*gaz)./(p.m.*p.V);

%Ориентация (углы Эйлера)
Dx.gamma = x.wx - cos(x.gamma).*tan(x.theta).*x.wy + sin(x.gamma).*tan(x.theta).*x.wz;
Dx.theta = sin(x.gamma).*x.wy + cos(x.gamma).*x.wz;
Dx.psi = (cos(x.gamma)./cos(x.theta)).*x.wy - (sin(x.gamma)./cos(x.theta)).*x.wz;

%Модель приводов, с ограничениями
Dx.stab = min(max(x.dstab, -p.maxabsdstab), p.maxabsdstab);
control_stab = min(max(u.stab, -p.maxabsstab), p.maxabsstab);

Dx.dstab = (-2.*p.Tstab.*p.Xistab.*x.dstab - x.stab + control_stab)./(p.Tstab.^2);
Dx.ail = min(max(x.dail, -p.maxabsdail), p.maxabsdail);

control_ail = min(max(u.ail, -p.maxabsail), p.maxabsail);

Dx.dail = (-2.*p.Tail.*p.Xiail.*x.dail - x.ail + control_ail)./(p.Tail.^2);
Dx.dir = min(max(x.ddir, -p.maxabsddir), p.maxabsddir);
control_dir = min(max(u.dir, -p.maxabsdir), p.maxabsdir);

Dx.ddir = (-2.*p.Tdir.*p.Xidir.*x.ddir - x.dir + control_dir)./(p.Tdir.^2);
Dx = cell2mat(struct2cell(Dx));

end

