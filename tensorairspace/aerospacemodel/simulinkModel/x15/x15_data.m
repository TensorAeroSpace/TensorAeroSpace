function [A, B, C, D] = x15_data()

g = 32.1740;
m = 15560 / g;
iy = 80003;

Mwd = 0;
Zwd = 0;
Zq = 0;
con = Mwd / (m - Zwd);

Xu = -0.00871;
Zu = 0.0117;
Mu = 0.000471;

Xw = -0.0190;
Zw = -0.311;
Mw = -0.00673 + (Mwd * Zq);

U0 = 1936 + Zq;
Mq = -0.182 + (Mwd * Zq);

Zde = -89.2;

Xd = 6.24 / m;
Zd = Zde / (m + Zwd);
Md = (-9.8 + Zde * con) / iy;

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