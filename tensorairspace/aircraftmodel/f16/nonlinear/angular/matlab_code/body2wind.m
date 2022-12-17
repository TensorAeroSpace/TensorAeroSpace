function [ V, alpha, beta ] = body2wind( Vx, Vy, Vz )
%Переход от связанной к скоростной системе координат
V = sqrt(Vx.^2 + Vy.^2 + Vz.^2);
alpha = -atan(Vy ./ Vx);
beta = asin(Vz ./ V);
end

