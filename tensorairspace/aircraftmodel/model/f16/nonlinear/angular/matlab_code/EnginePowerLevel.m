function dPa = EnginePowerLevel( Pa, throttle_act )
% --------------------------------------------------------------
%               Модель двигателя.
% --------------------------------------------------------------
% dPa = EnginePowerLevel(Pa, throttle_act) вычисляет скорость изменения уровня тяги двигателя в зависимости от
%    текущего уровня тяги и управляющего сигнала
%
% Pa - текущий уровень тяги, %
% throttle_act - управляющий сигнал, [0; 1]
% --------------------------------------------------------------

ntraj = size(Pa, 2);

if (throttle_act <= 0.77)
    Pc = 64.94 * throttle_act;
else
    Pc = 217.38 * throttle_act - 117.38;
end

for i = 1 : ntraj
    if (Pc(i) >= 50 && Pa(i) < 50)
        Pc(i) = 60;
    elseif (Pc(i) < 50 && Pa(i) >= 50)
        Pc(i) = 40;
    end
end

dP = Pc - Pa;

w_eng = zeros(1, ntraj);
for i = 1 : ntraj
    if (dP(i) <= 25)
        w_eng(i) = 1.0;
    elseif (dP(i) >= 50)
        w_eng(i) = 0.1;
    else
        w_eng(i) = 1.9 - 0.036 * dP(i);
    end
end

for i = 1 : ntraj
    if (Pa(i) >= 50)
        w_eng(i) = 5;
    end
end

dPa = w_eng.*dP;
