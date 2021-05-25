# diabetes-prediction
predict diabetes in Pima Indian people using neural network

we have Pima Indians diabetes dataset. and we want to predict whether a peron with some features has diabetes or not.
so this is a binary classification problem, we use neural network with 3 layers to predict that. there was 8 featurs(inputs) including: 
1.Number of times pregnant 
2.Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
3.Diastolic blood pressure (mm Hg).
4.Triceps skinfold thickness (mm).
5.2-Hour serum insulin (mu U/ml).
6.Body mass index (weight in kg/(height in m)^2).
7.Diabetes pedigree function.
8.Age (years).

and we design a 15 activations in hidden layer.
finally using FP and BP algorithm and advanced optimaziation we got 80% accuracy in training set.
you can also see learning curve and validation curve for checking bias , variance and lambda(regularization parameter)

you shoul run <strong>main</strong> in your octave command window
