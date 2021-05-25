function [J grad] = costFunction(params,input_layer_size,...
                     hidden_layer_size,output_layer_size,X,y,lambda)
                     
Theta1 = reshape(params(1:hidden_layer_size * (input_layer_size+1)),...
    hidden_layer_size,(input_layer_size+1));
    
Theta2 = reshape(params((1+ (hidden_layer_size * (input_layer_size+1))):end),...
       output_layer_size,(hidden_layer_size+1));

m = size(X,1);

J=0;  % costFunction value

% theta1&2 gradients
Theta1_grad = zeros(size(Theta1));   
Theta2_grad = zeros(size(Theta2));

%compute costFunction using FP algorithm
X1 = [ones(m,1) X];
a2 = sigmoid(X1 * Theta1');   %activation for hidden layer
mm = size(a2,1);
a2 = [ones(mm,1) a2];
h = sigmoid(a2 * Theta2');  % hypothesis function 

y_k = zeros(m,output_layer_size);
for i=1:m
  y_k(i,y(i))=1;
end

J = sum(sum( y_k.*log(h) + (1-y_k).*log(1-h)))/(-m);
J = J+ (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)) )* lambda/(2*m);
%================================================

% now the BP algorithm for gradients

for t=1:m
  a_1 = X(t,:)';
  a_1 = [1;a_1];
  z_2 = Theta1 * a_1;
  
  a_2 = sigmoid(z_2);
  a_2 = [1;a_2];
  z_3 = Theta2 * a_2;
  
  a_3 = sigmoid(z_3);
  
  y_t = y_k(t,:)';
  delta_3 = a_3 - y_t;
  
  delta_2 = (Theta2' * delta_3) .* sigmoidPrime([1;z_2]);
  delta_2 = delta_2(2:end);
  
  d_big_2 = delta_3 * a_2';
  d_big_1 = delta_2 * a_1';
  
  Theta1_grad += d_big_1;
  Theta2_grad += d_big_2;
end

Theta1_grad = (1/m) * Theta1_grad;
Theta2_grad = (1/m) * Theta2_grad;

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) * Theta2(:,2:end);

grad = [Theta1_grad(:);Theta2_grad(:)];

end