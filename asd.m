D = 6000;
N = 10e3;
Lw = 3;
Lt = 4;

y = rand(D,N);
muyk = rand(D,N);
covyk = diag(rand(D,1));
Awk = rand(D,Lw);
pre_Gammawk = rand(Lw,Lw);
Gammawk = pre_Gammawk'*pre_Gammawk;

[logr_old, a, b] = loggausspdf(y,muyk,covyk+Awk*Gammawk*Awk'); % Nx1
[logr, a1, b1] = loggausspdf_diag_lowk(y,muyk,diag(covyk),Awk,Gammawk); % Nx1