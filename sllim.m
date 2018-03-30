function [theta, r, LLf] = sllim(tapp, yapp, in_K, varargin)
  %%%%%%%% General EM Algorithm for Gaussian Locally Linear Mapping %%%%%%%%%
  %%% Author: Antoine Deleforge (April 2013) - antoine.deleforge@inria.fr %%%
  % Description: Compute maximum likelihood parameters theta and posterior
  % probabilities r=p(z_n=k|x_n,y_n;theta) of a sllim model with constraints
  % cstr using N associated observations t and y.
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%% Input %%%%
  %- t (LtxN) % Training latent variables
  %- y (DxN)  % Training observed variables
  %- in_K (int)  % Initial number of components
  % <Optional>
  %- Lw (int) % Dimensionality of hidden components (default 0)
  %- maxiter (int) % Maximum number of iterations (default 100)
  %- in_theta (struct)% Initial parameters (default [])
  %| same structure as output theta
  %- in_r (NxK)  % Initial assignments (default [])
  %- cstr (struct) % Constraints on parameters theta (default [],'')
  %- cstr.ct  % fixed value (LtxK) or ''=uncons.
  %- cstr.cw  % fixed value (LwxK) or ''=fixed to 0
  %- cstr.Gammat% fixed value (LtxLtxK) or ''=uncons.
  %| or {'','d','i'}{'','*','v'} (1)
  %- cstr.Gammaw% fixed value (LwxLwxK) or ''=fixed to I
  %- cstr.pi  % fixed value (1xK) or ''=uncons. or '*'=equal
  %- cstr.A  % fixed value (DxL) or ''=uncons.
  %- cstr.b  % fixed value (DxK) or ''=uncons.
  %- cstr.Sigma% fixed value (DxDxK) or ''=uncons.
  %| or {'','d','i'}{'','*'} (1)
  %- verb {0,1,2}% Verbosity (default 1)
  %%%% Output %%%%
  %- theta  (struct)  % Estimated parameters (L=Lt+Lw)
  %- theta.c (LxK) % Gaussian means of X
  %- theta.Gamma (LxLxK) % Gaussian covariances of X
  %- theta.A (DxLxK)  % Affine transformation matrices
  %- theta.b (DxK) % Affine transformation vectors
  %- theta.Sigma (DxDxK) % Error covariances
  %- theta.gamma (1xK)% Arellano-Valle and Bolfarine's Generalized t
  %-  phi  (struct)% Estimated parameters
  %- phi.pi (1xK)  % t weights of X
  %- phi.alpha (1xK)  % Arellano-Valle and Bolfarine's Generalized t
  %- r (NxK)  % Posterior probabilities p(z_n=k|x_n,y_n;theta)
  %- u (NxK)  % Posterior probabilities E[u_n|z_n=k,x_n,y_n;theta,phi)
  %%% (1) 'd'=diag., 'i'=iso., '*'=equal for all k, 'v'=equal det. for all k
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % ======================Input Parameters Retrieval=========================
  [Lw, maxiter, in_theta, in_r, cstr, verb] = ...
  process_options(varargin,'Lw',0,'maxiter',100,'in_theta',[],...
    # 'in_r',[],'cstr',struct(),'verb',0);

end  % function
