function [A, B, C, D] = geosat_data()

A = [
    	0 1 0;
    	0.01036 0 0.7753;
        0 -0.1774 0;
];

B = [
    0;
    0;
    0.1512;
];

C=eye(3,3);

D=zeros(3,1);

end