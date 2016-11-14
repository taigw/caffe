clear;
w1=5;
w2=3;
alpha =15;
beta =0.15;
gamma=6;
d =[0.213052-0.0723512, -0.237708-0.00639659, -0.16503+0.492441, 1.414];

dsq = d(4)*d(4);
isq = d(1)*d(1)+d(2)*d(2)+d(3)*d(3);
bi= exp(-dsq/2/alpha/alpha -isq/2/beta/beta);
sp= exp(-dsq/2/gamma/gamma);

k = w1*bi+w2*sp;
dw1 = bi;
dw2 = sp;
dalpha = w1*bi*dsq/(alpha*alpha*alpha);
dbeta = w1*bi*isq/(beta*beta*beta);
dgamma= w2*sp*dsq/(gamma*gamma*gamma);