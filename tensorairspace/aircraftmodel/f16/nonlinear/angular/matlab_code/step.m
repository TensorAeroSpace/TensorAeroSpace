function state = step(prev_state, dt, u, t0, i, parameters)
    % prev_step - предидущее состояние
    % dt - шаг дискретизации
    % u - управляющий сигнал
    % t0 - начальное время
    % i - текущий шаг системы
    % parameters - параметры самолета
    state = prev_state + dt * F16ODE(prev_state, u, t0 + dt * i, parameters);
end
