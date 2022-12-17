function [ xs ] = F16State_vec2struct( xv )
    xnames = {'alpha', 'wz', 'stab', 'dstab'};
    xs = cell2struct(con2seq(xv')', xnames);
end
