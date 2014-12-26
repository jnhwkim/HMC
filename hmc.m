function [ H ] = hmc( U, grad_U, epsilon, L, current_q )
%HMC SIMPLE IMPLEMENTATION OF HAMILTONIAN MONTE CARLO.
%
% Radford M. Neal, 2010.
%
% Ported to Matlab and modified by
% Jin-Hwa Kim, 2014. 
%
% Below description was originally written by Neal.
%
% This program appears in Figure 2 of "MCMC using Hamiltonian dynamics",
% to appear in the Handbook of Markov Chain Monte Carlo.
%
% The arguments to the HMC function are as follows:
%
%   U          A function to evaluate minus the log of the density of the
%              distribution to be sampled, plus any constant - ie, the
%              "potential energy".
%
%   grad_U     A function to evaluate the gradient of U.
%
%   epsilon    The stepsize to use for the leapfrog steps.
%
%   L          The number of leapfrog steps to do to propose a new state.
%
%   current_q  The current state (position variables only).
%
% Momentum variables are sampled from independent standard normal
% distributions within this function.  The value return is the vector
% of new position variables (equal to current_q if the endpoint of the
% trajectory was rejected).
%
% This function was written for illustrative purposes.  More elaborate
% implementations of this basic HMC method and various variants of HMC
% are available from my web page, http://www.cs.utoronto.ca/~radford/

  q = current_q;
  p = rand(length(q),1);  % independent uniform distribution
  current_p = p;

  % Make a half step for momentum at the beginning
  p = p - epsilon * grad_U(q) / 2;

  % Alternate full steps for position and momentum
  for i = 1 : L
    % Make a full step for the position
    q = q + epsilon * p;

    % Make a full step for the momentum, except at end of trajectory
    if i~=L 
        p = p - epsilon * grad_U(q);
    end
  end

  % Make a half step for momentum at the end.
  p = p - epsilon * grad_U(q) / 2;

  % Negate momentum at end of trajectory to make the proposal symmetric
  p = p; % nope

  % Evaluate potential and kinetic energies at start and end of trajectory
  current_U = U(current_q);
  current_K = 0;
  proposed_U = U(q);
  proposed_K = 0;

  % Accept or reject the state at end of trajectory, returning either
  % the position at the end of the trajectory or the initial position
  
  % fprintf('U0: %.3f, U1: %.3f, K0: %.3f, K1: %.3f\n', current_U,proposed_U,current_K,proposed_K);

  if rand(1) < min(1, exp(current_U-proposed_U+current_K-proposed_K))
    H = q;  % accept
    % fprintf('accept: %.4f\n', min(1, exp(current_U-proposed_U+current_K-proposed_K)));
    % fprintf('U0: %.3f, U1: %.3f, K0: %.3f, K1: %.3f\n', current_U,proposed_U,current_K,proposed_K);
  else
    H = current_q;  % reject
    % fprintf('reject: %.4f\n', min(1, exp(current_U-proposed_U+current_K-proposed_K)));
  end
end
