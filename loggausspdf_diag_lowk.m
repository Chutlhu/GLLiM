function [y, c, q] = loggausspdf_diag_lowk(X, mu, A, C, B)

[d,n] = size(X);
Lw = size(B,1);

if (sum(A == 0)>0)
    fprintf(1,'SNPD! ');
    y=-Inf(1,n); % 1xn
    return;
end

Xdiff = bsxfun(@minus,X,mu); % dxn difference

% WOODBURY MATRIX IDENTITY
% inversion of diagonal matrix
invA_diag = 1./A; % Dx1
D = inv(B) + bsxfun(@times,C,invA_diag)'*C; % LwxLw

% 1x1 NORMALIZATION CONSTANT
% logdetSigma = sum(log(A)) + log(det(eye(Lw) + C'*diag(invA_diag)*(C*B)));
logdetSigma = sum(log(A)) + log( ...
                                det(eye(Lw) ...
                                + bsxfun(@times, C, invA_diag)'*(C*B)));
c = d*log(2*pi)+logdetSigma;

bar = zeros(1,n);
for i=1:n
    a = (Xdiff(:,i).*invA_diag)'*C;
    bar(i) = (a/D)*a';
end

% % TODO: with broadcasting
% asd = bsxfun(@times, invA_diag,C);
% asd = (asd/D)*asd';
% bar4 = dot(Xdiff'*asd,Xdiff');


b = sum(bsxfun(@times, invA_diag, Xdiff.^2),1); % 1xN
q = b - bar;

y = -(c+q)/2; % 1xn

end