function [ signal ] = random_steps( t0, dt, tn, pd1, pd2 )
    n = floor((tn - t0) / dt) + 1;
    signal = zeros(1, n);
    step_start_time = t0;
    step_duration = random(pd1, 1, 1);
    step_value = random(pd2, 1, 1);
    for i = 1 : n
        t = t0 + i * dt;
        signal(i) = step_value;
        if t >= step_start_time + step_duration
            step_start_time = t;
            step_duration = random(pd1, 1, 1);
            step_value = random(pd2, 1, 1);
        end    
    end
end
