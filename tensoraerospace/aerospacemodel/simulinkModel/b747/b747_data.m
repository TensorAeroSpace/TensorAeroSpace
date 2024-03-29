function [A, B, C, D] = b747_data(flag)
%
%   1 - продольныйй канал
%   2 - боковой канал
%

if flag == 1
    A = [
        -0.0069	0.0139     	0   -9.8100;
        -0.0905	-0.3149  235.8928    0;
         0.0004	-0.0034   -0.4282     0;
     	 0     	0    1.0000     	0;
    ];

    B = [
        -0.0001;  
        -5.5079 ;    	
        -1.1569 ;    	
     	0 ;   
    ];
    
    C=eye(4,4);
    
    D=zeros(4,1);
end

if flag == 2
    A = [
        -0.1007 -0.2810 9.81 -157.5570;
        -0.0176 -0.8766 0.0 0.5754;
        0.0     1.0     0.0 0.0;
        0.0050  -0.069  0.0 -0.2810;
    ];

    B = [
        0;
        -.0540;
        0.0;
        -.3746;
    ];
    
    C=eye(4,4);
    
    D=zeros(4,1);   
end

end