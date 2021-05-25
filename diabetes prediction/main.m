data = load("diab.csv");
X = data(:,1:end-1); y = data(:,end);     % X is the feature matrix & y is the outputs
for i=1:size(y,1)
  if (y(i) == 0)      % 1 means he/she has diabetes but 2 means he/she doesnt have ones
    y(i)=2;
  endif
end

X_train = X(1:537,:); y_train= y(1:537,:);
X_cv = X(538:end,:); y_cv = y(538:end,:);

input_layer_size = 8;
hidden_layer_size = 15;                   % network architecture
output_layer_size = 2;

init_theta_1 = randInit(input_layer_size,hidden_layer_size);
init_theta_2 = randInit(hidden_layer_size,output_layer_size);

init_theta = [init_theta_1(:) ; init_theta_2(:)];

lambda = 1;
maxIter = 400;

options = optimset("maxIter",maxIter);

costFunc = @(p) costFunction(p,input_layer_size,...
            hidden_layer_size,output_layer_size,X_train,y_train,lambda);
         
[params , cost] = fmincg(costFunc,init_theta,options);

theta1 = reshape(params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
               
theta2 = reshape(params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 output_layer_size, (hidden_layer_size + 1));
                 
printf("\ncomputing Learning curve (lambda = %f)\n\n",lambda);
figure(1);
[error_train , error_cv] = biasVariance(X_train,y_train,X_cv,y_cv,lambda); 
plot(1:20, error_train, 1:20, error_cv);
title(sprintf('Learning Curve (lambda = %f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 21 0 20])
legend('Train', 'Cross Validation')     
fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:20
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_cv(i));
end     

printf("\ncomputing best lambda using validation curve\n\n");
[lambda_vec, error_train, error_cv] = ...
    cvCurve(X_train, y_train, X_cv, y_cv);
figure(2);
plot(lambda_vec, error_train, lambda_vec, error_cv);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_cv(i));
end    
                 
[pred p] = predict(theta1,theta2,X_train);
fprintf('\nTraining Set Accuracy: %f\n\n', mean(double(p == y_train)) * 100);

[pred p] = predict(theta1,theta2,X_cv);
fprintf('\nCross validation Set Accuracy: %f\n\n', mean(double(p == y_cv)) * 100);

fprintf("for example a person with these features has diabetes\n(9,119,80,35,0,29.0,0.263,29)\n");
[pred p] = predict(theta1,theta2,[9,119,80,35,0,29.0,0.263,29]);
fprintf('\nthe prediction value is: %f\n\n',pred *100);