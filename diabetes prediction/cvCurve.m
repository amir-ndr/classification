function [lambda_vec, error_train, error_cv] = ...
    cvCurve(X_train, y_train, X_cv, y_cv)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 0.7 1 3 5 10]';

error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

for i = 1:length(lambda_vec)
  X_train_ = X_train(1:i, :);
  y_train_ = y_train(1:i);
  
  init_theta_1 = randInit(8,15);
  init_theta_2 = randInit(15,2);
  init_theta = [init_theta_1(:) ; init_theta_2(:)];
  maxIter = 400;
  options = optimset("maxIter",maxIter);
  costFunc = @(p) costFunction(p,8,...
              15,2,X_train_,y_train_,lambda_vec(i));
           
  [params , cost] = fmincg(costFunc,init_theta,options);
  
  error_train(i)  = costFunction(params,8,15,2,X_train_,y_train_,0);
  error_cv(i)    = costFunction(params,8,15,2,X_cv,y_cv,0);
end
end