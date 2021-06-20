function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

# JTheta = (1/m Sum m = 1 to m, (-yi*log(hTheta(xi)) - (1-yi)*log(1-hTheta(xi))))
#          + (lambda / (2*m)) Sum j = 1 to n, thetaj^2
# Grad = 1/m Sum m = 1 to m, (hTheta - yi) * xj                      For j = 0
# Grad = 1/m Sum m = 1 to m, (hTheta - yi) * xj + (lambda/m) *thetaj For j >= 1

hTheta = sigmoid(theta' * X');
n = length(theta);
sumTheta = 0;
for j = 2:n
  sumTheta += theta(j)^2;
end

for i = 1:m
  J += ((-y(i) * log(hTheta(i)) - (1 - y(i)) * log(1 - hTheta(i)))) / m;
       
  grad += (((hTheta(i) - y(i)).*X(i,:))') / m;  
end

J += (lambda / (2 * m)) * sumTheta;

for k = 2:n
  grad(k) += (lambda / m) * theta(k);
endfor


% =============================================================

end
