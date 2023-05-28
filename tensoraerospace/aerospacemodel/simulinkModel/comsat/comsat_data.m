function [A, B, C, D] = comsat_data()

A = [
    	0 1 0;
    	0.01036 0 0.7757;
        0 -0.1775 0;
];

B = [
    0;
    0;
    0.1513;
];

C=eye(3,3);

D=zeros(3,1);

end