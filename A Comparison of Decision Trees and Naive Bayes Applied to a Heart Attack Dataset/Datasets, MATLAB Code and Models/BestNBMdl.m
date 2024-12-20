%This is the best naive bayes model 
%Lets first load the model and the X and Y test set
load('BestNBModel.mat');
load('X test set.mat');
load('Y test set.mat');

%lets find the training error
NBMdl_H_trainError_HP = resubLoss(NBMdl_HP);
%Lets make our predictions using the X test set
predictedlabels_NB_HP = predict(NBMdl_HP , X_test);

%MODEL EVALUATIONS - NAIVE BAYES (WITH HYPERPARAMETERS)
%We calculate the True Positives and Negatives and also the False Positives
%and Negatives
NB_confusion_matrix_HP = confusionmat(Y_test,predictedlabels_NB_HP);
TP_NB_HP = NB_confusion_matrix_HP(2,2);
TN_NB_HP = NB_confusion_matrix_HP(1,1);
FP_NB_HP = NB_confusion_matrix_HP(1,2); 
FN_NB_HP = NB_confusion_matrix_HP(2,1); 

Accuracy_NB_HP = (TP_NB_HP+TN_NB_HP) / (TP_NB_HP+TN_NB_HP+FP_NB_HP+FN_NB_HP);  
Precision_NB_HP = TP_NB_HP/(TP_NB_HP+FP_NB_HP);
Recall_NB_HP = TP_NB_HP/(TP_NB_HP+FN_NB_HP);
F1_Score_NB_HP = (2*TP_NB_HP) / ((2*TP_NB_HP)+FP_NB_HP+FN_NB_HP);
[~,pp_score_NB_HP] = predict(NBMdl_HP,X_test); 
pp_scores_positive_NB_HP = pp_score_NB_HP(:,2); 
[X_Rate_NB_HP,Y_Rate_NB_HP,~,AUC_NB_HP] = perfcurve(Y_test,pp_scores_positive_NB_HP,1);
%Reference link for how to calculate the above metrics:
%https://moodle4.city.ac.uk/mod/folder/view.php?id=382059

disp(['Accuracy of best NB model is: ' num2str(Accuracy_NB_HP)]);
disp(['Precision of best NB model is: ' num2str(Precision_NB_HP)]);
disp(['Recall of best NB model is: ' num2str(Recall_NB_HP)]);
disp(['F1 Score of best NB model is : ' num2str(F1_Score_NB_HP)]);
disp(['AUC Score of best NB model is : ' num2str(AUC_NB_HP)]);
%The close the AUC is to 1, the better.
%Now lets evaluate the results to see how well the model performed
%MODEL EVALUATIONS - NAIVE BAYES (HOLDOUT CROSS VALIDATION) with
%hyperparameters
confusion_matrix_NB_HP = confusionmat(Y_test,predictedlabels_NB_HP);

%Lets a create a confusion chart to see the distribution of TP, FP, TN and
%FN
figure;
confusion_chart_NB_HP = confusionchart(Y_test,predictedlabels_NB_HP);
title('NB with hyperparameters (Holdout)');
%Reference link for confusion matrix and chart: https://uk.mathworks.com/help/stats/confusionmat.html?fbclid=IwAR1ybVZQjwP_KYFCIfOg08QHPLhuYSxPE3Gm10q67UyOQosXLuboJJ_yFr0

%Lets plot the ROC curve
figure;
plot(X_Rate_NB_HP,Y_Rate_NB_HP,'b');
%Reference link for ROC curve: https://uk.mathworks.com/help/stats/perfcurve.html?fbclid=IwAR3yt-8iUsEGtWlTPCUUjT3vRf3_W3hwLmSNB47gqQyN68yUCbKZ_61ifkU#bupy9b3-1
%In the section: Plot ROC Curve for Classification Tree.
