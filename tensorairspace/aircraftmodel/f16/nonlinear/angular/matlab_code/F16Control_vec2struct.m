function [ us ] = F16Control_vec2struct( uv )
xnames = {'stab', 'ail', 'dir'};
us = cell2struct(con2seq(uv')', xnames);
end

