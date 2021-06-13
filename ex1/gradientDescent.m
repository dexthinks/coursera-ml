function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    sumTheta0 = 0;
    sumTheta1 = 0;
    for j = 1:m
      hx = theta(1) + theta(2) * X(j, 2);
      sumTheta0 += hx - y(j);
      sumTheta1 += (hx - y(j)) * X(j, 2);
    endfor
    theta(1) = theta(1) - (alpha/m) * sumTheta0;
    theta(2) = theta(2) - (alpha/m) * sumTheta1;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
