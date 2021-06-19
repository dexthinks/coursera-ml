function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

# JTheta = 1/m Sum m = 1 to m, (-yi*log(hTheta(xi)) - (1-yi)*log(1-hTheta(xi)))
# Grad = 1/m Sum m = 1 to m, (hTheta - yi) * xj

hTheta = sigmoid(theta' * X');
for i = 1:m
  J += ((-y(i) * log(hTheta(i)) - (1 - y(i)) * log(1 - hTheta(i)))) / m;
  grad += (((hTheta(i) - y(i)).*X(i,:))') / m;
end


% =============================================================

end
