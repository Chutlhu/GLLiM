function y = loggausspdf_diag(X, mu, Sigma_diag)

[d,n] = size(X);

if (sum(Sigma_diag == 0)>0)
    fprintf(1,'SNPD! ');
    y=-Inf(1,n); % 1xn
    return;
end

Xdiff = abs(bsxfun(@minus,X,mu)).^2; % dxn Squared difference
Xdiff = bsxfun(@rdivide,Xdiff,Sigma_diag); %dxn

c = d*log(2*pi)+sum(log(Sigma_diag)); % 1x1 normalization constant

q = sum(Xdiff,1); % 1*n

y = -(c+q)/2; % 1xn

end