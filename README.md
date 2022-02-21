# Credit_Risk_Analysis

## Overview :
The purpose of this analysis is to understand how to utilize Machine Learning statistical algorithms to make predictions based on data patterns provided. In this challenge, we focus on Supervised Learning using a free dataset from LendingClub, a P2P lending service company to evaluate and predict credit risk. The reason why this is called "Supervised Learning" is because the data includes a labeled outcome.
For this project I am utilizing several models of supervised machine learning on credit loan data in order to predict credit risk. Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. I will be using Python and Scikit-learn libraries and several machine learning models to compare the strengths and weaknesses of ML models and determine how well a model classifies and predicts data.

## Results:
In this analysis I used six different algorithms of supervised machine learning. First four algorithms are based on resampling techniques and are designed to deal with class imbalance. After the data is resampled, Logistic Regression is used to predict the outcome. Logistic regression predicts binary outcomes. The last two models are from ensemble learning group. The concept of ensemble learning is the process of combining multiple models, like decision tree algorithms, to help improve the accuracy and robustness, as well as decrease variance of the model, and therefore increase the overall performance of the model.

### The First 4 supervised Machine Learning Models from Resample Techniques are as follows :

**1.  Naive Random Oversampling and Logistic Regression:**

In random oversampling, instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced.

- Balanced Accuracy Score :

![image](https://user-images.githubusercontent.com/92283185/155030164-8b4d3a2f-eeee-452f-99c7-827acc0779ea.png)


- Precision & Recall Scores:

![image](https://user-images.githubusercontent.com/92283185/155029919-d7eb59a3-ec1c-4376-8eaf-642bd89109ff.png)

**2. SMOTE Oversampling:**

The synthetic minority oversampling technique (SMOTE) is another oversampling approach to deal with unbalanced datasets. In SMOTE, like random oversampling, the size of the minority is increased. The key difference between the two lies in how the minority class is increased in size. That is, for an instance from the minority class, a number of its closest neighbors is chosen. Based on the values of these neighbors, new values are created.

- Balanced Accuracy Score :

![image](https://user-images.githubusercontent.com/92283185/155031239-b511ef28-a3cf-467b-ae2d-95687c966206.png)

- Precision & Recall Scores:

![image](https://user-images.githubusercontent.com/92283185/155031273-9279e4a8-62c4-4b0e-b128-5264e9cd60ed.png)

**3. Undersampling - ClusterCentroids resampler :**

Undersampling takes the opposite approach of oversampling. Instead of increasing the number of the minority class, the size of the majority class is decreased.
Cluster centroid undersampling is akin to SMOTE. The algorithm identifies clusters of the majority class, then generates synthetic data points, called centroids, that are representative of the clusters. The majority class is then undersampled down to the size of the minority class.

- Balanced Accuracy Score:

![image](https://user-images.githubusercontent.com/92283185/155031774-6d2ef5cc-ed1e-4dd9-a102-4e923f5473e8.png)

- Precision & Recall Scores:

![image](https://user-images.githubusercontent.com/92283185/155031817-6328c8a6-184c-4468-ad80-e8f180703b1d.png)

**4. Combination (Over and Under) Sampling - SMOTEENN:**

SMOTEENN is an approach to resampling that combines aspects of both oversampling and undersampling - oversample the minority class with SMOTE and clean the resulting data with an undersampling strategy.

- Balanced Accuracy Score:

![image](https://user-images.githubusercontent.com/92283185/155032034-ae681266-9372-4a4c-a774-ab309882800b.png)

- Precision & Recall Scores:

![image](https://user-images.githubusercontent.com/92283185/155032063-ae2bf216-f570-4329-b5af-121914fe1dc1.png)

## Ensemble Technique:
The concept of ensemble learning is the process of combining multiple models, like decision tree algorithms, to help improve the accuracy and robustness, as well as decrease variance of the model, and therefore increase the overall performance of the model.

### The last 2 supervised Machine Learning Models from Ensemble Techniques are as follows:

**1. Balanced Random Forest Classifier:**

Instead of having a single, complex tree like the ones created by decision trees, a random forest algorithm will sample the data and build several smaller, simpler decision trees. Each tree is simpler because it is built from a random subset of features.

- Balanced Accuracy Score:

![image](https://user-images.githubusercontent.com/92283185/155032686-5ba1bebf-dc7d-442d-bcf6-506a60a2916a.png)

- Precision & Recall Scores:

![image](https://user-images.githubusercontent.com/92283185/155032739-0d702a36-3f79-4aaf-87a0-7cf4006130f1.png)

**2. Easy Ensemble AdaBoost Classifier:**

In AdaBoost, a model is trained then evaluated. After evaluating the errors of the first model, another model is trained. This time, however, the model gives extra weight to the errors from the previous model. The purpose of this weighting is to minimize similar errors in subsequent models. Then, the errors from the second model are given extra weight for the third model. This process is repeated until the error rate is minimized.

- Balanced Accuracy Score:

![image](https://user-images.githubusercontent.com/92283185/155032879-7a74c9df-04b1-4992-82e8-0011d5f2fe0d.png)

- Precision & Recall Score:

![image](https://user-images.githubusercontent.com/92283185/155032922-fc419148-4d53-49ed-b63a-aaecd05272fa.png)

## Summary:
All the models used to perform the credit risk analysis show weak precision in determining if a credit risk is high. The Ensemble models brought a lot more improvment specially on the sensitivity of the high risk credits. The EasyEnsembleClassifier model shows a recall of 93% so it detects almost all high risk credit. On another hand, with a low precision, a lot of low risk credits are still falsely detected as high risk which would penalize the bank's credit strategy and infer on its revenue by missing those business opportunities. For those reasons I would not recommend the bank to use any of these models to predict credit risk.



