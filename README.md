# Google Store Revenue Prediction
This work describes our approach and results on a Kaggle competition "Google Analytics Customer Revenue Prediction" by exploring the performance of different existing machine learning algorithms. Multiple machine learning models is built up, after preprocessing and feature engineering of the original data set. Model evaluation and assessment is illustrated in order to select the model as well as the corresponding hyperparameters. The best model and comparison of different models, and the preprocessing method is discussed.

## Introduction
The 80/20 rule, also known as the Pareto principle, states that roughly 80\% of the effects come from 20\% of the causes for many events.
In the view of business, the  80/20 rule means most of the sales are produced by only a small percent of customers. Therefore, it would be very important for marketing teams if they can make appropriate investments in promotional strategies by targeting their customers who 
can potentially lead to more sales. "Google Analytics Customer Revenue Prediction" is a competition available on Kaggle, posted by RStudio, Google Cloud and Kaggle aiming for discovering the possibility of providing suggestion to marketing teams through machine learning. The goal of this project is to analyze a Google Merchandise Store customer dataset to predict revenue per customer. Specifically, given information describing a customer like the geography, user device etc., we are supposed to fit regression on the revenue for each customer.

Many kernels are available on Kaggle associated with this competition. Most kernels illustrate the procedure of converting json columns and extensive contents on EDA. Among the machine learning models applied in these kernels, GBM is a very popular one, implemented by the lightGBM package. However, few kernels compare the effect of different preprocessing strategy and different models. We, therefore, studied this project more systematically and provided more insights by comparing different preprocessing methods and algorithm.

## Overview of Dataset
The dataset was split into training set and test set. (see Table \ref{table:TrainTest}) While the original data set only contains 12 features, we discovered four of them are json type data, which contains some sub-columns. 60 features presented after flatting json colums.
There are also some missing values for some features. The distribution of the missing value is listed in Table

## Exploratory Data Analysis
[Exploratory Data Analysis](https://github.com/ziwei1992/Google-Store-Revenue-Prediction/blob/master/Eda.ipynb): EDA EDA EDA

## Preprocessing
[Preprocessing 1](https://github.com/ziwei1992/Google-Store-Revenue-Prediction/blob/master/Preprocessing-1.ipynb):

[Preprocessing 2](https://github.com/ziwei1992/Google-Store-Revenue-Prediction/blob/master/Preprocessing-2.ipynb):

[Preprocessing 3](https://github.com/ziwei1992/Google-Store-Revenue-Prediction/blob/master/Preprocessing-3.ipynb):

## Machine Learning Models
[Linear Regression/Ridge/Lasso]():

[Random Forest]():

[LGBM]():

[ANN]():

## Conclusions




