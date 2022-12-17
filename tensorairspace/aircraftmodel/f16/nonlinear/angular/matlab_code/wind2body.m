function [ Vx, Vy, Vz ] = wind2body( V, alpha, beta )
%Переход от скоростной к связанной системе координат
Vx = V .* cos(alpha) .* cos(beta);
Vy = -V .* sin(alpha) .* cos(beta);
Vz = V .* sin(beta);
end

