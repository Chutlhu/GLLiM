Ds = [100, 1000, 3000];
Ns = [100, 1000, 10000];

Lw = 2;
Lt = 4;
time_old = zeros(12,1);
time = zeros(12,1);
i = 1;

for N=Ns
    for D=Ds
        y = rand(D,N);
        muyk = rand(D,N);
        covyk = diag(rand(D,1));
        Awk = rand(D,Lw);
        pre_Gammawk = rand(Lw,Lw);
        Gammawk = pre_Gammawk'*pre_Gammawk;

        tic
        [logr_old, a, b] = loggausspdf(y,muyk,covyk+Awk*Gammawk*Awk'); % Nx1
        time_old(i) = toc;
        tic
        [logr, a1, b1] = loggausspdf_diag_lowk(y,muyk,diag(covyk),Awk,Gammawk); % Nx1
        time(i) = toc;
        i = i+1;
    end
end

plot([time, time_old])