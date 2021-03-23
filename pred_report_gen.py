# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 23:12:04 2021

@author: Shreevanth Gopalakrishnan
"""
# ENSEMBLE PREDICTION AND REPORT GENERATION

######################################################################
# MODULE/PACKAGE IMPORT

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import os
import shutil
from joblib import dump, load
from scipy.signal import savgol_filter


######################################################################
# CUSTOM FUNCTIONS

def import_scaler(input_path):
# Function to import the training data used, and import the fitted scaler

    file_X = os.path.join(input_path, 'X_train.csv')
    file_y = os.path.join(input_path, 'y_train.csv')
    
    # Only importing the central 6 features
    dataset = pd.read_csv(file_X)
    X = dataset.iloc[:, 3:9].values
    X2 = dataset.iloc[:,3 + 10: 9 + 10].values
    X = np.concatenate((X,X2), axis = 1)
    
    # Included for completeness, but does not affect scaling
    dataset = pd.read_csv(file_y)
    y = dataset.iloc[:, 1].values
    y = np.reshape(y, (np.shape(y)[0], 1))
    
    # Shuffle dataset for robustness
    dataset = np.concatenate([X, y], axis = 1)
    dataset = shuffle(dataset)
    
    X = dataset[:,:12]
    y = dataset[:,12]
    
    # Create and fit a feature scaler
    sc_X = StandardScaler()
    sc_X.fit(X) 

    return sc_X


def angles(dataset):
# Function to calculate the RHS Wrist-Elbow-Shoulder angle

    #######################
    # Angle 1: R_U_Angle
    
    # Numerator
    prod_1 = (dataset['x_RWrist'] - dataset['x_RElbow']) * \
        (dataset['x_RElbow'] - dataset['x_RShoulder']) 

    prod_2 =(dataset['y_RWrist'] - dataset['y_RElbow']) * \
        (dataset['y_RElbow'] - dataset['y_RShoulder'])
    
    d_p1 = prod_1 + prod_2
    
    # Denominator
    mag_1 = np.sqrt((dataset['x_RWrist'] - dataset['x_RElbow'])**2 + \
        (dataset['y_RWrist'] - dataset['y_RElbow'])**2)
        
    mag_2 = np.sqrt((dataset['x_RElbow'] - dataset['x_RShoulder'])**2 + \
        (dataset['y_RElbow'] - dataset['y_RShoulder'])**2)
    
    mag1 = mag_1 * mag_2
    
    #######################
    # Angle 2: L_U2_Angle
    
    # Numerator
    prod_3 = (dataset['x_Neck'] - dataset['x_LShoulder']) * \
        (dataset['x_LElbow'] - dataset['x_LShoulder']) 

    prod_4 = (dataset['y_Neck'] - dataset['y_LShoulder']) * \
        (dataset['y_LElbow'] - dataset['y_LShoulder'])
    
    # Dot product = a1*a2 + b1*b2 (only parallel components are multiplied)
    d_p2 = prod_3 + prod_4
    
    # Denominator
    mag_3 = np.sqrt((dataset['x_Neck'] - dataset['x_LShoulder'])**2 + \
        (dataset['y_Neck'] - dataset['y_LShoulder'])**2)
        
    mag_4 = np.sqrt((dataset['x_LElbow'] - dataset['x_LShoulder'])**2 + \
        (dataset['y_LElbow'] - dataset['y_LShoulder'])**2)
    
    mag2 = mag_3 * mag_4
    
    #######################
    # Final calculations for R_U_Angle
    R_U_Angles = np.zeros(len(d_p1))
    
    for ii in range(len(d_p1)):
        try:
            result1 = d_p1[ii]/mag1[ii]
        except ZeroDivisionError:
            result1 = 0
           
        if result1 > 1:
            result1 = 1
        
        if math.isnan(result1):
            result1 = 1
    
        R_U_Angle = np.arccos(result1) * 180/(math.pi)
        R_U_Angles[ii] = 180 - R_U_Angle
        
    R_U_Angles = np.reshape(np.array(R_U_Angles),(len(R_U_Angles),1))
    # R_U_Angles = np.reshape(R_U_Angles, (np.shape(R_U_Angles)[0],1))
    
    #######################   
     # Final calculations for L_U2_Angle
     
    L_U2_Angles = np.zeros(len(d_p2))
     
    for ii in range(len(d_p2)):
       try:
           result2 = d_p2[ii]/mag2[ii]
       except ZeroDivisionError:
           result2 = 0
           
       if result2 > 1:
           result2 = 1
      
       if result2 <(-1):
           result2 = -1
        
       if math.isnan(result2):
           result2 = 1
   
       L_U2_Angle = np.arccos(result2) * 180/(math.pi)
      
            
       L_U2_Angles[ii] = L_U2_Angle  
    
    L_U2_Angles = np.reshape(np.array(L_U2_Angles),(len(L_U2_Angles),1))
    
    return R_U_Angles, L_U2_Angles

def normalize_centre(Angle1,Angle2):
    
    fin_Angle1 = np.zeros((10,1))
    fin_Angle2 = np.zeros((10,1))
    
    # Normalizing    
    max_rom1 = np.amax(Angle1)
    max_rom2 = np.amax(Angle2)
    
    Angle1 = (max_rom1 - Angle1)/max_rom1
    Angle2 = (max_rom2 - Angle2)/max_rom2
    
    # Centering such that the peak of R_U_Angle occurs in the middle
    max_rom1 = np.amax(Angle1)
    
    pos = np.where(Angle1 == max_rom1)[0]
    
    # If there are duplicates, pick the first occurrence
    if pos.size == 2:
        pos = pos[0]
    
    # # Reshaping into a row vector
    # R_U_Angle = np.reshape(R_U_Angle, \
    #                            (np.shape(R_U_Angle)[0],1))
    
    # Centre both the angles based on only the R_U_Angle
    fin_Angle1[5] = max_rom1
    fin_Angle2[5] = Angle2[pos]
    
    try: 
        # First half
        idx1 = np.linspace(0,pos-1,5)
        idx1 = idx1.astype(int)
        idx1 = idx1.flatten()
        fin_Angle1[0:5,0] = Angle1[idx1] 
        fin_Angle2[0:5,0] = Angle2[idx1] 
        
        # Second half
        idx2 = np.linspace(pos+1, np.size(Angle1)-2,4)
        idx2 = idx2.astype(int)
        idx2 = idx2.flatten()
        fin_Angle1[6:10,0] = Angle1[idx2] 
        fin_Angle2[6:10,0] = Angle2[idx2]
        
    except IndexError:
        # If there is considerable error, then do not bother
        idx = np.linspace(0, np.size(Angle1)-1,10)
        idx = idx.astype(int)
        idx2 = idx2.flatten()
        fin_Angle1[:,0] = Angle1[idx] 
        fin_Angle2[:,0] = Angle2[idx] 
        
    except ValueError:
        print('Value Error')
    
    # Reshaping into a row vector
    fin_Angle1 = np.reshape(fin_Angle1, \
                               (1,np.shape(fin_Angle1)[0]))
        
    fin_Angle2 = np.reshape(fin_Angle2, \
                               (1,np.shape(fin_Angle2)[0]))    
    
    return fin_Angle1, fin_Angle2


def process_test_data(input_path):
# Function to load and pre-process the test data into a single .csv file
# 'input_path' must specify where the reps videos are
    
    IN_PATH = os.path.join(input_path,'keypoints')
    OUT_PATH = os.path.join(input_path,'X_test.csv')
    ERROR_PATH = ('D:/00 - Cranfield University MSc Work/09 - \
                  Group Project/CDR/Shree_Source/Test_data/Error_videos')
    all_data = os.listdir(IN_PATH)
    Angles_full = pd.DataFrame(columns = ['id_rep','R1','R2','R3','R4','R5','R6',\
                                     'R7','R8','R9','R10','L1','L2','L3','L4',\
                                         'L5','L6','L7','L8','L9','L10'])

    for ii in range(len(all_data)):
        
        with open(os.path.join(IN_PATH, all_data[ii]),'r') as data:
            dataset = pd.read_csv(data)
        (R_U_Angle,L_U2_Angle) = angles(dataset)
        R_U_Angle = pd.DataFrame(R_U_Angle, columns = ['R_U_Angle'])
        L_U2_Angle = pd.DataFrame(L_U2_Angle, columns = ['L_U2_Angle'])
        Angles = pd.concat([R_U_Angle, L_U2_Angle], axis = 1, ignore_index = True)
        Angles = np.array(Angles) 
        # Angles = np.reshape(Angles, (np.shape(Angles)[0],))
        
        try:
            smoothened_1 = savgol_filter(Angles[1:,0], 5, 2)
            smoothened_2 = savgol_filter(Angles[1:,1], 5, 2)
            
        except ValueError:
            print('Number of keypoints in the video is too few.', all_data[ii])
            # Move file to a diff. folder if the rep counting was erroneous
            shutil.move(os.path.join(IN_PATH, all_data[ii]), \
                        os.path.join(ERROR_PATH, all_data[ii]))
            continue
        
        (R_U_Angle,L_U2_Angle) = normalize_centre(smoothened_1, \
                                              smoothened_2)
        
        Angle = np.concatenate((R_U_Angle,L_U2_Angle), axis = 1) 
        Angle = pd.DataFrame(Angle, columns = ['R1','R2','R3','R4','R5','R6',\
                                     'R7','R8','R9','R10','L1','L2','L3','L4',\
                                         'L5','L6','L7','L8','L9','L10'])
        
        Angle.insert(0, "id_rep", all_data[ii], True)     
        Angles_full = Angles_full.append(Angle, ignore_index = True)    
        Angles_full.to_csv(OUT_PATH, index = False)
    
    
def load_test_data(input_path, sc_X):
# Function to load the processed test data into memory and identify different \
# ... athletes

    IN_PATH = os.path.join(input_path, 'X_test.csv')  
    
    # Reading the test dataset and sorting athlete by athlete (alphabetic)
    dataset = pd.read_csv(IN_PATH)
    dataset = dataset.sort_values(by=['id_rep'])
    
    # Separating the most relevant 6 central features of the athlete's pull-up
    X_test = dataset.iloc[:, 3:9].values
    X2 = dataset.iloc[:,3 + 10: 9 + 10].values
    X_test = np.concatenate((X_test,X2), axis = 1)
    
    # Applying the pre-fitted standard scaler to the test set
    X_test = sc_X.transform(X_test) 
    
    # # In case, it is supervised: read in the ground truth test labels
    # IN_PATH = os.path.join(input_path, 'y_test.csv') 
    # dataset = pd.read_csv(IN_PATH)
    # dataset = dataset.sort_values(by=['video_id'])
    # y_test = dataset.iloc[:, 1].values
    
 
    # Identifying different athletes IDs based on the filename
    athletes = dataset.iloc[:, 0].values
    fname = []
    ctr = 0
    reps = 0
    athlete = {}
    
    for ii in range(len(athletes)):        

        fname = athletes[ii]
        fname = ''.join(ii for ii in fname if not ii.isdigit())
        fname = fname.replace('_', '')
        fname = fname.replace('.csv', '')
        
        if ii == 0:
            reps = reps + 1
            athlete[fname] = reps
            ctr = 1
            continue
        
        list_keys = list()
        for i in athlete.keys():
            list_keys.append(i)
         
        # Checking if the repetition counting has started for any athlete    
        check_athlete = any(key in fname for key in list_keys)    
         
        if check_athlete == 1:
            reps = reps + 1
            athlete[fname] = reps
            
        elif check_athlete == 0:
            ctr = ctr + 1
            reps = 1
            athlete[fname] = reps
                
    return X_test, athlete

def load_all_clf(input_path):
# Function to import all pre-trained classifiers

    clfs = []  
    
    # Loading only the relevant classifier files from the file system
    for file in os.listdir(input_path):
        if file.endswith('.joblib'):
            print('Found the {:s} classifier!'.format(file[:-7]))
            file = os.path.join(input_path, file)
            clfs.append(file)
            
    # Loading the 5 classifiers separately
    clf1 = load(str(clfs[0]))
    clf2 = load(str(clfs[1]))
    clf3 = load(str(clfs[2]))
    clf4 = load(str(clfs[3]))
    clf5 = load(str(clfs[4]))
    
    return clf1, clf2, clf3, clf4, clf5

def predictions(clf1, clf2, clf3, clf4, clf5, X_test):
# Function to make predictions on the test set

    print('--------------------------')
    print('**INFO: Test set predictions being made..')

    #######################
    # Prediction with Classifier 1
    y_pred1_prob = clf1.predict_proba(X_test)
    y_pred1 = clf1.predict(X_test)
    # acc1 = accuracy_score(y_test, y_pred1)*100
    # cm1 = confusion_matrix(y_test, y_pred1)
    # model = 'AdaptiveBoostedCART' 
    # print(model)
    # print(acc1)
    # print(cm1)
    # print('--------------------------')
    
    #######################
    # Prediction with Classifier 2
    y_pred2_prob = clf2.predict_proba(X_test)
    y_pred2 = clf2.predict(X_test)
    # acc2 = accuracy_score(y_test, y_pred2)*100
    # cm2 = confusion_matrix(y_test, y_pred2)
    # model = 'AdaptiveBoostedKSVM' 
    # print(model)
    # print(acc2)
    # print(cm2)
    # print('--------------------------')
    
    #######################
    # Prediction with Classifier 3
    y_pred3_prob = clf3.predict_proba(X_test)
    y_pred3 = clf3.predict(X_test)
    # acc3 = accuracy_score(y_test, y_pred3)*100
    # cm3 = confusion_matrix(y_test, y_pred3)
    # model = 'RandomForest' 
    # print(model)
    # print(acc3)
    # print(cm3)
    # print('--------------------------')
    
    #######################
    # Prediction with Classifier 4
    y_pred4_prob = clf4.predict_proba(X_test)
    y_pred4 = clf4.predict(X_test)
    # acc4 = accuracy_score(y_test, y_pred4)*100
    # cm4 = confusion_matrix(y_test, y_pred4)
    # model = 'BaggedKernelSVM' 
    # print(model)
    # print(acc4)
    # print(cm4)
    # print('--------------------------')
    
    #######################
    # Prediction with Classifier 5
    y_pred5_prob = clf5.predict_proba(X_test)
    y_pred5 = clf5.predict(X_test)
    # acc5 = accuracy_score(y_test, y_pred5)*100
    # cm5 = confusion_matrix(y_test, y_pred5)
    # model = 'KNearestNeighbors' 
    # print(model)
    # print(acc5)
    # print(cm5)
    # print('--------------------------')


    #######################
    # Majority voting and confidence interval estimation
    preds = np.zeros((len(X_test),5))
    votes = np.zeros(len(X_test))
    mean = np.zeros(len(X_test))
    stddev = np.zeros(len(X_test))
    res = np.zeros(len(X_test))   
    conf = np.zeros(len(X_test))   
    
    for ii in range(len(X_test)):
       preds[ii,0] = y_pred1[ii]  
       preds[ii,1] = y_pred2[ii]
       preds[ii,2] = y_pred3[ii]  
       preds[ii,3] = y_pred4[ii]  
       preds[ii,4] = y_pred5[ii]
       votes[ii] = np.count_nonzero(preds[ii,:])
             
       # Voting for class 0: Incorrect rep
       if votes[ii] < 3:
           temp = 0
       # Voting for class 1: Correct rep
       else:
           temp = 1
           
       # Storing all the probabilities into one array
       probs = np.array([y_pred1_prob[ii,temp], y_pred2_prob[ii,temp], \
                         y_pred3_prob[ii,temp], y_pred4_prob[ii,temp], \
                             y_pred5_prob[ii,temp]])    
       
       # Calculating the mean and standard deviation of the predictions
       n = len(probs)
       mean[ii] = sum(probs)/n
       var = sum((x - mean[ii]) ** 2 for x in probs)/n
       stddev[ii] = math.sqrt(var)
       conf[ii] = mean[ii] - stddev[ii]
       res[ii] = temp
       
    print('**INFO: Test set predictions are complete.')   
    print('--------------------------')
    
    return res, conf


def report_gen(res, conf, athlete_db):
# Function to generate a performance report for this call of the clf block
# Additional functionality: Ability to specifically highlight reps predicted 
# ... with low confidence (mean - stddev)
    
    print('--------------------------')
    print('**INFO: Generating performance report')
    
    # Retrieving the names of all athletes, and their number of reps
    list_keys = list()
    ctr = 0
    pos = 0
    
    # Counting the number of correct and incorrect reps, and the confidence
    # ... in their prediction 
    correct = np.zeros(len(athlete_db.keys()))
    incorrect = np.zeros(len(athlete_db.keys()))
    low_conf = np.zeros(len(athlete_db.keys()))
    
    for ii in athlete_db.keys():
        list_keys.append(ii)
        reps = athlete_db.get(str(ii))
        correct[ctr] = np.count_nonzero(res[pos:pos + reps]) 
        print('--------------------------')
        string = 'The athlete {} carried out {} correct reps out of {}'.\
            format(str(ii),int(correct[ctr]),int(reps))
        print(string)
        incorrect[ctr] = reps - correct[ctr]
        low_conf[ctr] = np.count_nonzero(conf[pos:pos + reps] < 0.65) 
        string = 'Of those, {} have been classified with low confidence.'.\
            format(int(low_conf[ctr]))
        print(string)
        print('--------------------------')
        
        plt.figure(ctr+1)
        plt.pie([correct[ctr],incorrect[ctr]], \
                labels=['correct', 'incorrect'], startangle=90)
        plt.title('Athlete: {:s}'.format(str(ii)))
        ctr = ctr + 1
        pos = pos + reps

        


######################################################################
# Using an ensemble of the top 5 models to predict probability and therefore, 
# provide a confidence interval

######################################################################
# SECTION 1: LOAD SCALER FROM TRAINING PHASE

# Specify relative path of training data
data_path = 'Data/'

sc_X = import_scaler(data_path)

######################################################################
# SECTION 2: PRE-PROCESS & LOAD TEST DATA (PRE-SPLIT INTO REPS & IN .csv)

# Specify relative path of test data
data_path = 'Test_data/'

# Creating a compatible X_test.csv file
process_test_data('pull_up_data')

# Loading all the test data and athlete information
(X_test,athlete_db) = load_test_data('pull_up_data',sc_X)
print(X_test)
######################################################################
# SECTION 3: IMPORT CLASSIFIERS

# Specify relative path of all trained classifier models
clf_path = 'Trained_models/'
(clf1,clf2,clf3,clf4,clf5) = load_all_clf(clf_path)

######################################################################
# SECTION 4: MAKE PREDICTIONS

# Making predictions for each for the samples in X_test, irrespective 
# ... of the athlete in question
(res,conf) = predictions(clf1,clf2,clf3,clf4,clf5,X_test)

######################################################################
# SECTION 5: GENERATE PERFORMANCE REPORT

# As per the predictions made, plot the performance of different athletes

report_gen(res, conf, athlete_db)
