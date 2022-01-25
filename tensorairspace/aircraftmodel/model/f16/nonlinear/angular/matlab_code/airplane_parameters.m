function [ parameters ] = airplane_parameters( )
% --------------------------------------------------------------
%       Значения по умолчанию для постоянных параметров.
% --------------------------------------------------------------
%Oy - высота над уровнем моря, м
%V - воздушная скорость, м/с
%q - скоростной напор, Па
%m - масса самолета, кг
%l - размах крыла, м
%S - площадь крыла, м^2
%bA - САХ, м
%Jx, Jy, Jz - моменты инерции относительно осей связанной системы координат, м/с^2
%Jxy, Jyz, Jxz - центробежные моменты инерции, м/с^2
%rcgx - смещение от центра тяжести до аэродинамического фокуса по продольной оси, м
%hEx - момент импульса двигателя, кг м^2/с
%Tstab, Tail, Tdir - постоянные времени приводов
%Xistab, Xiail, Xidir - коэффициенты относительного демпфирования приводов
%maxabsstab, maxabsail, maxabsdir, minthrottle, maxthrottle - ограничения на величины управляющих сигналов, рад
%maxabsdstab, maxabsdail, maxabsddir - ограничения на угловые скорости управляющих поверхностей, рад/с
%lef - отклонение носков крыла, рад
%sb - отклонение воздушного тормоза, рад
%g - ускорение свободного падения на поверхности Земли, м/с^2
% --------------------------------------------------------------
parameters.m = 9295.44;%kg
parameters.l = 9.144;%m
parameters.S = 27.87;%m^2
parameters.bA = 3.45;%m
parameters.Jx = 12874.8;%m/sec^2
parameters.Jy = 85552.1;%m/sec^2
parameters.Jz = 75673.6;%m/sec^2
parameters.Jxy = 1331.4;%m/sec^2
parameters.Jyz = 0;%m/sec^2
parameters.Jxz = 0;%m/sec^2
parameters.rcgx = -0.05 * parameters.bA; %m
parameters.hEx = 0.0;

parameters.Tstab = 0.03;
parameters.Xistab = 0.707;

parameters.maxabsstab = deg2rad(25);
parameters.maxabsdstab = deg2rad(60);
parameters.Tail = 0.02;
parameters.Xiail = 0.707;

parameters.maxabsail = deg2rad(21.5);
parameters.maxabsdail = deg2rad(80);
parameters.Tdir = 0.03;
parameters.Xidir = 0.707;

parameters.maxabsdir = deg2rad(30);
parameters.maxabsddir = deg2rad(120);

parameters.lef = 0;
parameters.sb = 0;
parameters.g = 9.80665;

parameters.Oy = 3000;
parameters.V = 120;
%Модель атмосферы ISA (для высоты до 11000 м над уровнем моря)
L = 0.0065;%скорость уменьшения температуры с увеличением высоты, К/м
R = 287.0531;%удельная газовая постоянная (для сухого воздуха), Дж/(кг К)
T0 = 288.15;%температура воздуха на уровне моря, К
T = T0 - L * parameters.Oy;%температура воздуха на текущей высоте, К
rho0 = 1.225;%плотность воздуха на уровне моря, кг/м^3
rho = rho0 * (T / T0).^(parameters.g / (L * R) - 1);%плотность воздуха на текущей высоте, кг/м^3
parameters.q = (rho.*(parameters.V.^2)) / 2;%скоростной напор, Па

end

