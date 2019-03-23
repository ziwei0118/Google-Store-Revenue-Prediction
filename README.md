# Google Store Revenue Prediction
This work describes our approach and results on a Kaggle competition "Google Analytics Customer Revenue Prediction" by exploring the performance of different existing machine learning algorithms. Multiple machine learning models is built up, after preprocessing and feature engineering of the original data set. Model evaluation and assessment is illustrated in order to select the model as well as the corresponding hyperparameters. The best model and comparison of different models, and the preprocessing method is discussed.

#### Table of Contents
**[Introduction](#Introduction)**<br>
**[Overview of Dataset](#Overview-of-Dataset)**<br>
**[General workflow](#General-workflow)**<br>
**[Exploratory Data Analysis](#Exploratory-Data-Analysis)**<br>
**[Preprocessing](#Preprocessing)**<br>
**[Machine Learning Models](#Machine-Learning-Models)**<br>
**[Comparison between models/preprocessing methods](#Comparison-between-models/preprocessing-methods)**<br>
**[Conclusions](#Conclusions)**<br>

## Introduction
The 80/20 rule, also known as the Pareto principle, states that roughly 80\% of the effects come from 20\% of the causes for many events.
In the view of business, the  80/20 rule means most of the sales are produced by only a small percent of customers. Therefore, it would be very important for marketing teams if they can make appropriate investments in promotional strategies by targeting their customers who 
can potentially lead to more sales. "Google Analytics Customer Revenue Prediction" is a competition available on Kaggle, posted by RStudio, Google Cloud and Kaggle aiming for discovering the possibility of providing suggestion to marketing teams through machine learning. The goal of this project is to analyze a Google Merchandise Store customer dataset to predict revenue per customer. Specifically, given information describing a customer like the geography, user device etc., we are supposed to fit regression on the revenue for each customer.

Many kernels are available on Kaggle associated with this competition. Most kernels illustrate the procedure of converting json columns and extensive contents on EDA. Among the machine learning models applied in these kernels, GBM is a very popular one, implemented by the lightGBM package. However, few kernels compare the effect of different preprocessing strategy and different models. We, therefore, studied this project more systematically and provided more insights by comparing different preprocessing methods and algorithm.

## Overview of Dataset
The dataset was split into training set and test set. While the original data set only contains 12 features, we discovered four of them are json type data, which contains some sub-columns. 60 features presented after flatting json colums (see the table below). There are also some missing values for some features. Many features are catergorical values. 

| Dataset  | Samples | Features (original) | Features (converting json) |
| ------------- | ------------- | ------------- | ------------- |
| Training  | 1708337  | 12  | 59  |
| Test  | 401589  | 12  | 59  |

## General workflow
The following steps are performed to explory and make prediction on this dataset: Data cleaning, Exploratory Data Analysis, Preprocessing, train machine learning models, deploy the best model.

## Exploratory Data Analysis
In this section, we will show our initial data cleaning and Exploratory Data Analysis. Please see the notebook for more details.

[Exploratory Data Analysis](https://github.com/ziwei1992/Google-Store-Revenue-Prediction/blob/master/Eda.ipynb): This notebook shows our initial data cleaning process and Exploratory Data Analysis.

## Preprocessing
In this section, we will show our data preprocessing based on previous Exploratory Data Analysis and important features selected from the models in the next section. Please see the notebooks for more details.

[Preprocessing 1](https://github.com/ziwei1992/Google-Store-Revenue-Prediction/blob/master/Preprocessing-1.ipynb): This is our initial preprocessing strategy (Impute 0 or 1 to the missing value, encode the categorical features, and merge the levels of catergorical feature. ).

[Preprocessing 2](https://github.com/ziwei1992/Google-Store-Revenue-Prediction/blob/master/Preprocessing-2.ipynb): Based on the feedback from LGBM, we selected the most important features after preprocessing method described in the first notebook.

[Preprocessing 3](https://github.com/ziwei1992/Google-Store-Revenue-Prediction/blob/master/Preprocessing-3.ipynb): We selected the most important features based on PCA after preprocessing method described in the first notebook.

## Machine Learning Models
[Linear Regression/Ridge/Lasso]():

[Random Forest]():

[LGBM]():

[ANN]():

## Comparison between models/preprocessing methods

## Conclusions

Prediction of revenue on customer is extremely important for marketing. In this work, we used the Google Merchandise Store customer dataset to predict revenue per customer. We first analyzed our dataset and did the EDA throughly. From the EDA, we noticed several challenges, like missing value, categorical data with too many levels. We also acquired the idea of how to reduce the data dimension in the preprocessing from EDA. Preprocessing enabled us to obtain a clean data, and allow us to vary the number of selected feature to compare how model performance depend on feature selection. Multiple algorithm were used to train the model, including LR, Lasso, Ridge, kNN, RF, GBM, ANN, DNN, Ada Boost Regression and ensemble models. We compared the performance of different models and found Ada Boost Regression has the best performance for this dataset. We also explored how feature selection can affect our models (LR, RF, GBM). 25 features selected by feature importance of GBM has the best performance, better than more or less features, and better than the features selected by PCA.

### Acknowledgements:
Some data in "Comparison between models" came from Wentao Yu and Fatemeh Amrollahi, my teammates in this Kaggle competition. I would like to thank them for their contribution.



