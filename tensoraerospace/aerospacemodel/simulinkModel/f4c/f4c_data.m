function [A, B, C, D] = f4c_data()

g = 32.1740;
m = 38925 / g;
iy = 122193;

Mwd = -0.0840 * 10^(-4);
Zwd = -0.000358;
Zq = -2.24;
con = Mwd / (m - Zwd);

Xu = -0.00679;
Zu = 0.0110;
Mu = 0.00341;

Xw = 0.00146;
Zw = -0.494;
Mw = -0.0198 + (Mwd * Zq);

U0 = 1472 - 2.24;
Mq = -0.488 + (Mwd * Zq);

Zde = -70.6;

Xd = 3.21 / m;
Zd = Zde / (m + Zwd);
Md = (-16 + Zde * con) / iy;

A = [
    Xu Xw 0 -g;
    Zu Zw U0 0;
    Mu Mw Mq 0;
     0 0 1.0000 0;
];

B = [
    Xd;  
    Zd;    	
    Md;    	
    0 ;   
];

C=eye(4,4);

D=zeros(4,1);


end