%This is the best decision trees model 
%Lets first load the model and the X and Y test set
load('BestDTModel.mat');
load('X test set.mat');
load('Y test set.mat');

%lets find the training error
DTMdl_H_trainError_HP = resubLoss(DTMdl_HP);
%Lets make our predictions using the X test set
predictedlabels_DT_HP = predict(DTMdl_HP , X_test);

%Now lets evaluate the results to see how well the model performed
%MODEL EVALUATIONS - DECISION TREES (HOLDOUT CROSS VALIDATION) with
%hyperparameters
confusion_matrix_DT_HP = confusionmat(Y_test,predictedlabels_DT_HP);

TP_DT_HP = confusion_matrix_DT_HP(2,2);
%True Positives: The model predicted correctly 
%the positive label '1' (has the disease)
TN_DT_HP = confusion_matrix_DT_HP(1,1); %True Negatives: The model predicted 
%correctly the negative label '0' (does not have the disease)
FP_DT_HP = confusion_matrix_DT_HP(1,2); %False Positives: The model predicted a 
%positive label '1' when the actual label is negative '0'
FN_DT_HP = confusion_matrix_DT_HP(2,1); %False Negative: The model predicted a 
%negative label '0' when the actual label is positive '1'

Accuracy_DT_HP = (TP_DT_HP+TN_DT_HP) / (TP_DT_HP+TN_DT_HP+FP_DT_HP+FN_DT_HP);  %Accuracy is a useful metric 
%when the dataset is balanced.
Precision_DT_HP = TP_DT_HP/(TP_DT_HP+FP_DT_HP);
Recall_DT_HP = TP_DT_HP/(TP_DT_HP+FN_DT_HP);
F1_Score_DT_HP = (2*TP_DT_HP) / ((2*TP_DT_HP)+FP_DT_HP+FN_DT_HP);
%Reference link for how to calculate the above metrics:
%https://moodle4.city.ac.uk/mod/folder/view.php?id=382059
[~,pp_score_DT_HP] = predict(DTMdl_HP,X_test); %The Predict function 
%transforms the label predictions into the posterior probability scores.
pp_scores_positive_DT_HP = pp_score_DT_HP(:,2); %These are the posterior 
%probability scores for the positive label '1' for each observation.
[X_Rate_DT_HP,Y_Rate_DT_HP,~,AUC_DT_HP] = perfcurve(Y_test,pp_scores_positive_DT_HP,1);
%In the above code we calculate the ROC graph and AUC using the perfcurve
%function. X_Rate is the False Positive and Y_Rate is the True Positive. 

disp(['Accuracy of best DT model is: ' num2str(Accuracy_DT_HP)]);
disp(['Precision of best DT model is: ' num2str(Precision_DT_HP)]);
disp(['Recall of best DT model is: ' num2str(Recall_DT_HP)]);
disp(['F1 Score of best DT model is : ' num2str(F1_Score_DT_HP)]);
disp(['AUC Score of best DT model is : ' num2str(AUC_DT_HP)]);
%The close the AUC is to 1, the better.



%Lets a create a confusion chart to see the distribution of TP, FP, TN and
%FN
figure;
confusion_chart_DT_HP = confusionchart(Y_test,predictedlabels_DT_HP);
title('DT with hyperparameters (Holdout)');
%Reference link for confusion matrix and chart: https://uk.mathworks.com/help/stats/confusionmat.html?fbclid=IwAR1ybVZQjwP_KYFCIfOg08QHPLhuYSxPE3Gm10q67UyOQosXLuboJJ_yFr0

%Lets plot the ROC curve
figure;
plot(X_Rate_DT_HP,Y_Rate_DT_HP,'b');
%Reference link for ROC curve: https://uk.mathworks.com/help/stats/perfcurve.html?fbclid=IwAR3yt-8iUsEGtWlTPCUUjT3vRf3_W3hwLmSNB47gqQyN68yUCbKZ_61ifkU#bupy9b3-1
%In the section: Plot ROC Curve for Classification Tree.