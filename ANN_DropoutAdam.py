#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 09:00:01 2018

@author: Ziwei Guo
"""

import numpy as np 
import pandas as pd 
import json
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import os
import tensorflow as tf

def multilayer_perceptron(x, weights, biases, keep_prob):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

def build_ANN(data_train,data_train_labels,test_x,test_y,n_hidden_units,drop_out_rate):
    n_hidden_1 = n_hidden_units
    n_input = data_train.shape[1] # number of featurs
    n_out_put= 1 # it's real value 
    
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_hidden_1, n_out_put]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_out_put]))
    }
    
    keep_prob = tf.placeholder(tf.float32)
    training_epochs = 500
    batch_size = 64 

    x = tf.placeholder(tf.float32,[None, n_input])
    y = tf.placeholder(tf.float32,[None, n_out_put])
    predictions=multilayer_perceptron(x,weights,biases,keep_prob)
    
    
    cost = tf.reduce_mean(tf.pow(tf.subtract(predictions,y),2))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
    Acc=0
    display_step=100 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(len(data_train) / batch_size)
            x_batches=(np.array_split(data_train, total_batch))
            y_batches = np.array(np.array_split(data_train_labels, total_batch))
            
            for i in range(total_batch):
                batch_x, batch_y = x_batches[i], y_batches[i]
                batch_x=np.reshape(batch_x,(len(batch_x),25))
                batch_y=np.reshape(batch_y,(len(batch_y),1))
            
                _,c = sess.run([optimizer,cost],feed_dict={x: batch_x,
                                            y: batch_y,keep_prob:float(1-drop_out_rate)})
                
                avg_cost += c/total_batch
                
            if epoch % display_step == 0:
                print("Epoch:",(epoch+1), "cost",avg_cost)
        
        test_x=np.reshape(batch_x,(len(batch_x),25))
        test_y=np.reshape(batch_y,(len(batch_y),1))
        Acc+=sess.run([cost],feed_dict={x: test_x, y: test_y, keep_prob:1.0})
        
    return Acc  


def main():
    df_train = pd.read_csv('./train_P2.csv')
    df_test = pd.read_csv('./test_P2.csv')
    
    df_train_y=df_train['totals.transactionRevenue']
    df_test_y=df_test['totals.transactionRevenue']
    
    df_train_x=df_train[['channelGrouping','fullVisitorId','visitNumber',
                     'customDimensions.value','device.browser',
                     'device.browser','device.deviceCategory','device.isMobile',
                     'device.operatingSystem','geoNetwork.city','geoNetwork.continent',
                     'geoNetwork.country','geoNetwork.metro','geoNetwork.region',
                     'totals.hits','totals.pageviews','totals.sessionQualityDim',
                     'totals.timeOnSite','trafficSource.referralPath','trafficSource.medium',
                     'trafficSource.source','weekday','year','day','month']]
    
    df_test_x=df_test[['channelGrouping','fullVisitorId','visitNumber',
                     'customDimensions.value','device.browser',
                     'device.browser','device.deviceCategory','device.isMobile',
                     'device.operatingSystem','geoNetwork.city','geoNetwork.continent',
                     'geoNetwork.country','geoNetwork.metro','geoNetwork.region',
                     'totals.hits','totals.pageviews','totals.sessionQualityDim',
                     'totals.timeOnSite','trafficSource.referralPath','trafficSource.medium',
                     'trafficSource.source','weekday','year','day','month']]
    
    data_train=df_train_x.values
    data_test=df_test_x.values
    data_train_labels=df_train_y.values
    data_test_labels=df_test_y.values
    #print train_x.shape
    n_fold=2
    fold_size=int(len(data_train) / n_fold)
    n_hidden_units=[25,50,12]
    drop_out_rates=[0.0,0.1,0.2,0.3,0.4,0.5]
    parameters_selections=[];
    for drop_out_rate in drop_out_rates:
        for n_h_u in n_hidden_units:
            Acc=0;
            for fold in range(n_fold):
                train_x = np.concatenate((data_train[0:fold*fold_size,:],data_train[fold*fold_size+fold_size:,:]))
                train_x=train_x.astype(float)
                valid_x= data_train[fold*fold_size:fold*fold_size+fold_size,:]
                valid_x=valid_x.astype(float)
                train_y=np.concatenate((data_train_labels[0:fold*fold_size],data_train_labels[fold*fold_size+fold_size:]))
                train_y=train_y.astype(float)
                valid_y = data_train_labels[fold*fold_size:fold*fold_size+fold_size]
                valid_y=valid_y.astype(float)
                Acc+=build_ANN(train_x,train_y,valid_x,valid_y,n_h_u,drop_out_rate)/n_fold
            parameters_selections.append([Acc,n_h_u,1-drop_out_rate])
    print parameters_selections
                
if __name__ == '__main__':
    main()
