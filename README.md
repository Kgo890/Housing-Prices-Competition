Context(why):
- Took on a Kaggle challenge where we used a preset database and test out machine learning basics 

Problem(question):
- You are looking at houses in the town of Ames, Iowa. You as a homeowner want to find the best valued property 
that typical home buyer would be looking for like when was it built, how big the lot is, etc. 
so for this problem, my job was to predict the sales price of each house or how they put it 
"for each ID in the test set, you must predict the value of the SalePrice variable"

Solution(answer):
- I first looked into both datasets (the training and testing dataset). then preprocessed the data for non-numeric features because Gradient Boosting can't work directly with
non-numeric data by using One-Hot Encoding making those categorical values into binart columns. As well as handling missing values and also normalizing the numerical features 
so that features with large ranges don't dominate smaller features when optimization. I then split the training dataset to 80% being used for training the models and 20% will be used for validation 
I then did a spotcheck for each algorithm that I used to see which one is performing the best. The models that I used was Linear Regression, Gradient Boosting Regressor, and Random Forest Regressor. 
I used those models because this problem is a regression problem and those are the one that I looked up would be good for this problem. 
I then evaluated each model using kfold and pipeline and used there R2 score to see which one is better. I also used box and whisker plot to compare algorithms 
After finding the best algorithm for the problem, I then evaluated predictions and then made predictions on the testing dataset. 
Finally, made a submission file to complete the challenge, that was formatted like this:
- ID, SalePrice
- 1461,129136.7401174314
- 1462,159600.11994747777
- 1463,182479.95184453798
- 1464,189899.88910744363
- 1465,188330.65480759877


Findings:
- what I found was that Gradient Booster Regressor was the best algorithm out of the 3 that I used having a higher R2 score then the rest being 85.66%
and a lower std of 0.0686 show that it performed more consistent across different data splits 
After using Gradient Booster Regressor predicting house prices on the validation set, the r2 score went up with it being 
86.36% and the Root Mean Squared Error is $ 31,187 
I founded that Gradient Booster Regressor could not use non-numerical values, so I had to preprocess the data to use it 

Limitations:
- My knowledge of some of Machine Learning concepts and problems because this is the first oen, so trying to improve the results is 
I don't know how to and also some of the information I don't know It's good or not

Conclusion (why+question+answer):
- So for my first Kaggle competition, the problem was to use Kaggle's datasets(training and testing) to find the sale price of each house in the dataset using their ID's and the SalePrice variables. 
So what I found was the SalePrice of each house and put it in a submission file using Gradient Boosting Regressor 
