function [A, B, C, D] = rocket_data()

A = [
    	-0.0089 -0.1474 0.0 -9.75;
    	-0.0216 -0.3601 5.9470 -0.151;
    	0.0 	-0.00015 -0.0224 0.0006;
    	0.0 0.0 1.0 0.0;
];

B = [
    	9.748;
    	3.77;
    	-0.034;
    	0.01;  
];

C=eye(4,4);

D=zeros(4,1);

end