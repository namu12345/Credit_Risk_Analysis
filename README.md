# Credit_Risk_Analysis

## Overview :
The purpose of this analysis is to understand how to utilize Machine Learning statistical algorithms to make predictions based on data patterns provided. In this challenge, we focus on Supervised Learning using a free dataset from LendingClub, a P2P lending service company to evaluate and predict credit risk. The reason why this is called "Supervised Learning" is because the data includes a labeled outcome.
For this project I am utilizing several models of supervised machine learning on credit loan data in order to predict credit risk. Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. I will be using Python and Scikit-learn libraries and several machine learning models to compare the strengths and weaknesses of ML models and determine how well a model classifies and predicts data.

## Results
In this analysis I used six different algorithms of supervised machine learning. First four algorithms are based on resampling techniques and are designed to deal with class imbalance. After the data is resampled, Logistic Regression is used to predict the outcome. Logistic regression predicts binary outcomes (1). The last two models are from ensemble learning group. The concept of ensemble learning is the process of combining multiple models, like decision tree algorithms, to help improve the accuracy and robustness, as well as decrease variance of the model, and therefore increase the overall performance of the model (6).

1.  Naive Random Oversampling and Logistic Regression:

In random oversampling, instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced (2).
