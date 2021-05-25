function [pred p] = predict(Theta1, Theta2, X)

m = size(X, 1);

p = zeros(size(X, 1), 1);
pred = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
[pred, p] = max(h2, [], 2);

% =========================================================================


end