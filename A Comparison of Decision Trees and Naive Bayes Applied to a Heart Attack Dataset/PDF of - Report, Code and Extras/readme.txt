For each part of my exploratory analysis, data preprocessing, splitting the data, model training and model evaluation the code is devided to sections. Each model is a different section of the code with the training of the model in the beginning and the evaluation of the model after. This will make it easier for you to run the code for each model separately. 

In my zip folder I have included the following files:
1. Heart_Disease_Dataset.csv. This csv file is the dataset that was obtained from Kaggle and renamed by me.
2. heart_attack.m . This is the full script of the Matlab code. Here we have all of the code
3. BestDTMdl.m. This is the script of the code only used for the best decision trees model. Here we have loaded the model and the X and Y test set to run the code and see the results.
4. BestNBMdl.m. This is the script of the code only used for the best naive bayes model. Here we have loaded the model and the X and Y test set to run the code and see the results.
5. BaselineDTModel.mat. This file represents the baseline decision trees model without any hyperparameters
6.BaselineNBModel.mat. This file represents the baseline naive bayes model without any hyperparameters
7. BestDTModel.mat. This .mat file represents the optimized decision trees model with hyperparameters
8. BestNBModel.mat. This .mat file represents the optimized naive bayes model with hyperparameters
9. test_set.csv. This csv file is the test set that was used to test and evaluate each model. It contains the observations for both the independent variables and the dependent variable that were selected for testing the models after applying holdout cross-validation. It also includes the headers. This test set was used for all models.
10. X test set.csv. This csv file contains the X test set used to test the models
11. Y test set.csv. This csv file contains the Y test set used to test the models
12. X test set.mat. This .mat file is the X test set that was loaded in the BestDTMdl.m and BestNBMdl.m to run the code for each model separately.
13. Y testset.mat. This .mat file is the Y test set that was loaded in the BestDTMdl.m and BestNBMdl.m to run the code for each model separately.
