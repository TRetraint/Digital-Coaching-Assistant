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
from numpy import genfromtxt
from numpy import percentile

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import make_scorer

import os
import shutil
from joblib import dump, load
from scipy.signal import savgol_filter

# Defining f1 score
f1_scorer = make_scorer(fbeta_score, beta=1, average = 'weighted')
######################################################################
# CUSTOM FUNCTIONS

def BinaryCrossentropy(y, yHat, res):
    
    ans = np.zeros(len(y))
    score = 0
    
    for ii in range(len(y)):
        
        if res[ii] == 1:
            y_pred = yHat[ii] 
        
        else:
            y_pred = 1 - yHat[ii]
        
        if y[ii] == 1:
            ans[ii] = -math.log(y_pred)
            
        else:
            ans[ii] = -math.log(1 - y_pred)

    score = sum(ans)/len(y)

    return score


def import_scaler(input_path):
# Function to import the training data used, and import the fitted scaler

    file_X = os.path.join(input_path, 'tvt_vectors.csv')
    file_y = os.path.join(input_path, 'tvt_labels.csv')
    
    # Only importing the central 6 features
    dataset = pd.read_csv(file_X)
    X = dataset.iloc[:,1:].values
    
    # Included for completeness, but does not affect scaling
    dataset = pd.read_csv(file_y)
    y = dataset.iloc[:, 1:].values

    # Shuffle dataset
    dataset = np.concatenate([X, y], axis = 1)
    dataset = shuffle(dataset)
    
    X = dataset[:,:24]
    y = dataset[:,24:]
    
    # Create and fit a feature scaler
    sc_X = StandardScaler()
    sc_X.fit(X) 

    return sc_X

###############################################################################
# CUSTOM FUNCTIONS

def pos(data, a, side):
# Function to extract magnitudes of certain joints from the dataset
    
    # For RHS, side = 'R'; for LHS, side = 'L'
    # Definitions for RHS/LHS
    a_x = 'x_' + side + a
    x = data[a_x]
    a_y = 'y_' + side + a
    y = data[a_y]
    
    return x, y


def vector(x_1, y_1, x_2, y_2):
# Function to calculate vector magnitude given vector start and end x, y's
   
    v = np.sqrt((x_2 - x_1)**2 + (y_2 - y_1)**2)

    return v


def calcs (data, a_1, a_2, side):
# Handler function to calculate the magnitude of the vector given joint name
# Careful with definitions of 2 and 1

    (x_2, y_2) = pos(data, a_1, side)
    (x_1, y_1) = pos(data, a_2, side)
    v = vector(x_1, y_1, x_2, y_2)

    return v


def dot(data, a_1, a_2, a_3, a_4, side):
# Function to calculate the dot product of two vectors, given the joint names
    
    (x_2, y_2) = pos(data, a_1, side)
    (x_1, y_1) = pos(data, a_2, side)
    
    (x_4, y_4) = pos(data, a_3, side)
    (x_3, y_3) = pos(data, a_4, side)
    
    x = (x_2 - x_1) * (x_4 - x_3)
    y = (y_2 - y_1) * (y_4 - y_3)
    
    res = x + y

    return res


def features(data):
# Function to derive features from the joint positions during the exercise 
    
    # Joints relevant for mistake Type-1 (T-1)
    a_1 = 'Wrist'
    a_2 = 'Elbow'
    a_3 = 'Elbow'
    a_4 = 'Shoulder'
    
    # Joints relevant for mistake Type-2 (T-2)
    a_5 = 'Hip'
    a_6 = 'Knee'
    a_7 = 'Knee'
    a_8 = 'Ankle'
      
    # Definitions for Upper chain (RHS)
    v_1 = calcs(data, a_1, a_2, 'R')
    v_2 = calcs(data, a_3, a_4, 'R')
    
    # Definitions for Upper chain (LHS)
    v_3 = calcs(data, a_1, a_2, 'L')
    v_4 = calcs(data, a_3, a_4, 'L')
    
    # Definitions for Lower Chain (RHS)
    v_5 = calcs(data, a_5, a_6, 'R')
    v_6 = calcs(data, a_7, a_8, 'R')
    
    # Definitions for Lower Chain (LHS)
    v_7 = calcs(data, a_5, a_6, 'L')
    v_8 = calcs(data, a_7, a_8, 'L')
    
    
    #######################
    # Upper chain (ARM) vector magnitudes on RHS and LHS
    side = ['R', 'L']
    
    for jj in side:
        
        # cos(theta) term: Numerator (dot product of the two vectors)
        dot_ = dot(data, a_1, a_2, a_3, a_4, jj)
        
        # Initialise vector to hold the vector magnitudes & other props
        if jj == 'R':
            R_U_Vectors = np.zeros(len(dot_))
            v_a = v_1
            v_b = v_2
            
        else:
            L_U_Vectors = np.zeros(len(dot_))
            v_a = v_3
            v_b = v_4
               
        # Calculation of cos(theta) = (A dot B)/(mag(A)*mag(B))
        for ii in range(len(dot_)):
            try:
                cos_ = dot_[ii]/(v_a[ii] * v_b[ii])
            except ZeroDivisionError:
                cos_ = 0
                
            if cos_ > 1:
                cos_ = 1
            
            if math.isnan(cos_):
                cos_ = 1
        
            # Final calc of the vector magnitude SQRT(A^2 + B^2 + 2ABcos(t))
            res = np.sqrt(v_a[ii]**2 + v_b[ii]**2 + 2*v_a[ii]*v_b[ii]*cos_)
            
            if jj == 'R':
                R_U_Vectors[ii] = res
            elif jj == 'L':
                L_U_Vectors[ii] = res


    #######################
    # Lower chain (LEG) vector magnitudes on RHS and LHS
    side = ['R', 'L']
    
    for jj in side:
        
        # cos(theta) term: Numerator (dot product of the two vectors)
        dot_ = dot(data, a_5, a_6, a_7, a_8, jj)
        
        # Initialise vector to hold the vector magnitudes & other props
        if jj == 'R':
            R_D_Vectors = np.zeros(len(dot_))
            v_a = v_5
            v_b = v_6
            
        else:
            L_D_Vectors = np.zeros(len(dot_))
            v_a = v_7
            v_b = v_8
        
        # Calculation of cos(theta) = (A dot B)/(mag(A)*mag(B))
        for ii in range(len(dot_)):
            try:
                cos_ = dot_[ii]/(v_a[ii] * v_b[ii])
            except ZeroDivisionError:
                cos_ = 0
                
            if cos_ > 1:
                cos_ = 1
            
            if math.isnan(cos_):
                cos_ = 1
        
            # Final calculation of the vector magnitude
            res = np.sqrt(v_a[ii]**2 + v_b[ii]**2 + 2*v_a[ii]*v_b[ii]*cos_)
            
            if jj == 'R':
                R_D_Vectors[ii] = res
            elif jj == 'L':
                L_D_Vectors[ii] = res
    
      
    return R_U_Vectors, L_U_Vectors, R_D_Vectors, L_D_Vectors


def clean_smooth(y, box_pts):
# Function to clean the dataset using the interquartile method & then smooth \
# using the Savitsky-Golay filter    
    
    #######################
    # Cleaning the dataset by the Interquartile range method
    
    q_25, q_75 = percentile(y, 25), percentile(y, 75)
    IQR = q_75 - q_25
    print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q_25, q_75, IQR))
    
    # Calculate outlier cutoff
    cut = IQR * 1.5
    lb, ub = q_25 - cut, q_75 + cut
    count = 0
    
    # Interpolate where outliers are identified
    for ii in range(len(y)):
        # If an outlier is detected: interpolate linearly between the adjacent
        # Accounting for ordinary case and identifications at extreme positions
        
        if y[ii] < lb or y[ii] > ub:
            outlier = True
            count = count + 1
        else:
            outlier = False
            
        if ii == 0 and outlier:
            y[ii] = y[ii+1]
        
        temp = len(y) - 1
        
        if ii == temp and outlier:
            y[ii] = y[ii-1]
            
        if ii > 0 and ii != temp and outlier:
            try:
                y[ii] = (y[ii-1] + y[ii+1])/2
            except IndexError:
                print('Index Error!')
                    
    print('Number of outliers:', count)
    #######################
    # Smoothing using the Savitsky-Golay filter
    
    y_smooth = savgol_filter(y, box_pts, 2)
    
    
    #######################
    # Reducing near-0 values to 0
    
    for ii in range(len(y_smooth)):
       if y_smooth[ii] < 1e-4:
           y_smooth[ii] = 0
    
    return y_smooth

def normalize_centre(Vector_1, Vector_2, Vector_3, Vector_4):
# Function to normalize the values of the vectors w.r.t. to the smoothened max.
# ... ,to centre the values into a 10-element column vector, and to sample the 
# central six features of each set
    
    # Initialise centering vectors
    fin_Vector_1 = np.zeros((10,1))
    fin_Vector_2 = np.zeros((10,1))
    fin_Vector_3 = np.zeros((10,1))
    fin_Vector_4 = np.zeros((10,1))
    
    #######################
    # Normalizing with the max value of each so that the range becomes [0,1]
    max_rom_1 = np.amax(Vector_1)
    max_rom_2 = np.amax(Vector_2)
    
    # Normalizing with the starting position (first 5 frames)
    max_rom_3 = np.amax(Vector_3[0:5])
    max_rom_4 = np.amax(Vector_4[0:5])
    
    Vector_1 = (max_rom_1 - Vector_1)/max_rom_1
    Vector_2 = (max_rom_2 - Vector_2)/max_rom_2
    Vector_3 = abs((max_rom_3 - Vector_3)/max_rom_3)
    Vector_4 = abs((max_rom_4 - Vector_4)/max_rom_4)
    
    #######################
    # Centering such that the trough of the upper chain vectors occur @ centre
    max_rom_1 = np.amax(Vector_1)
    max_rom_2 = np.amax(Vector_2)
    
    # Using the average index of the RHS and LHS max. values of upper chain
    pos_1 = np.where(Vector_1 == max_rom_1)[0]
    pos_2 = np.where(Vector_2 == max_rom_2)[0]
       
    # If there are duplicates, pick the first occurrence
    try:
        if np.shape(pos_1)[0] > 1:
            pos_1 = pos_1[0]
            
        if np.shape(pos_2)[0] > 1:
            pos_2 = pos_2[0]    
    
        pos = int((pos_1 + pos_2)/2)
    except TypeError:
        print('Type Error!')
        # pos_1_temp = np.where(Vector_1 == max_rom_1)[0]
        # pos_2_temp = np.where(Vector_2 == max_rom_2)[0]
        # if np.shape(pos_1_temp)[0] > 1:
        #     pos_1_temp = pos_1_temp[0]
            
        # if pos_2_temp.size == 2:
        #     pos_2_temp = pos_2_temp[0]    
    
        # pos_temp = int((pos_1_temp + pos_2_temp)/2)
        
    # Allocate min value of Vectors 1 & 2 to the central index (6/10)
    fin_Vector_1[5] = Vector_1[pos]
    fin_Vector_2[5] = Vector_2[pos]
    fin_Vector_3[5] = Vector_3[pos]
    fin_Vector_4[5] = Vector_4[pos]
    
    try: 
        # First half
        idx1 = np.linspace(0,pos-1,5)
        idx1 = idx1.astype(int)
        idx1 = idx1.flatten()
        fin_Vector_1[0:5,0] = Vector_1[idx1] 
        fin_Vector_2[0:5,0] = Vector_2[idx1] 
        fin_Vector_3[0:5,0] = Vector_3[idx1] 
        fin_Vector_4[0:5,0] = Vector_4[idx1] 
        
        # Second half
        idx2 = np.linspace(pos+1, np.size(Vector_1)-2,4)
        idx2 = idx2.astype(int)
        idx2 = idx2.flatten()
        fin_Vector_1[6:10,0] = Vector_1[idx2] 
        fin_Vector_2[6:10,0] = Vector_2[idx2]
        fin_Vector_3[6:10,0] = Vector_3[idx2] 
        fin_Vector_4[6:10,0] = Vector_4[idx2]
        
    except IndexError:
        # If there is considerable error, then do not bother
        idx = np.linspace(0, np.size(Vector_1)-1,10)
        idx = idx.astype(int)
        idx2 = idx2.flatten()
        fin_Vector_1[:,0] = Vector_1[idx] 
        fin_Vector_2[:,0] = Vector_2[idx] 
        fin_Vector_3[:,0] = Vector_3[idx] 
        fin_Vector_4[:,0] = Vector_4[idx] 
        
    except ValueError:
        print('Value Error')
    
    #######################
    # Sample the central six values of first two and returning to main function
    fin_Vector_1 = fin_Vector_1[2:8]   
    fin_Vector_2 = fin_Vector_2[2:8]
    fin_Vector_3 = fin_Vector_3[2:8]
    fin_Vector_4 = fin_Vector_4[2:8]
    
    
    # Reshaping into a row vector
    fin_Vector_1 = np.reshape(fin_Vector_1, \
                               (1,np.shape(fin_Vector_1)[0]))
        
    fin_Vector_2 = np.reshape(fin_Vector_2, \
                               (1,np.shape(fin_Vector_2)[0]))    
    
    fin_Vector_3 = np.reshape(fin_Vector_3, \
                               (1,np.shape(fin_Vector_3)[0]))
        
    fin_Vector_4 = np.reshape(fin_Vector_4, \
                               (1,np.shape(fin_Vector_4)[0])) 
        
    return fin_Vector_1, fin_Vector_2, fin_Vector_3, fin_Vector_4


def process_test_data(IN_PATH, OUT_PATH):
# Function to load and pre-process the test data into a single .csv file
# 'input_path' must specify where the reps videos are
  
    all_data = os.listdir(IN_PATH)

    # Only storing the six central features of each set for ease of use downstream
    Vectors = pd.DataFrame(columns = ['video_id','RU1','RU2','RU3','RU4','RU5','RU6',\
                                      'LU1','LU2','LU3','LU4','LU5','LU6',\
                                      'RD1','RD2','RD3','RD4','RD5','RD6',\
                                      'LD1','LD2','LD3','LD4','LD5','LD6'])
       
    for ii in range(len(all_data)):
        with open(os.path.join(IN_PATH, all_data[ii]),'r') as data:
            dataset = pd.read_csv(data)
        
        (R_U_Vectors,L_U_Vectors,R_D_Vectors,L_D_Vectors) = features(dataset) 
        
        smooth_upper_r = clean_smooth(R_U_Vectors,5)
        smooth_upper_l = clean_smooth(L_U_Vectors,5)
        smooth_lower_r = clean_smooth(R_D_Vectors,5)
        smooth_lower_l = clean_smooth(L_D_Vectors,5)
        
        
        (R_U_Vectors,L_U_Vectors,R_D_Vectors,L_D_Vectors) = \
            normalize_centre(smooth_upper_r,smooth_upper_l,\
                              smooth_lower_r,smooth_lower_l)
        
        Vector = np.concatenate((R_U_Vectors,L_U_Vectors,R_D_Vectors,L_D_Vectors),\
                                axis = 1)   
        
        Vector = pd.DataFrame(Vector, columns = ['RU1','RU2','RU3',\
                                                  'RU4','RU5','RU6','LU1','LU2',\
                                                  'LU3','LU4','LU5','LU6','RD1',\
                                                  'RD2','RD3','RD4','RD5','RD6',\
                                                  'LD1','LD2','LD3','LD4',\
                                                      'LD5','LD6'])
        
        Vector.insert(0, "video_id", all_data[ii], True)     
        Vectors = Vectors.append(Vector, ignore_index = True)
        
        OUT_FILE = OUT_PATH + all_data[ii]
        Vector.to_csv(OUT_FILE, index = False)
        Vectors.to_csv('Data/test_vectors.csv', index = False)
        
    
def load_test_data(input_path, sc_X):
# Function to load the processed test data into memory and identify different \
# ... athletes

    IN_PATH1 = os.path.join(input_path, 'test_vectors.csv')  
    IN_PATH2 = os.path.join(input_path, 'test_labels.csv')
    
    # Reading the test dataset and sorting athlete by athlete (alphabetic)
    X_test = pd.read_csv(IN_PATH1)
    X_test = X_test.sort_values(by=['video_id'])
    
    # Hidden ground truth test labels
    df = pd.read_csv(IN_PATH2)
    dataset = df.sort_values(by=['video_id'])
    y_test = dataset.iloc[:, 1:].values
    
    # y_test = np.reshape(y_test, (np.shape(y_test)[0],1))
    
    # # Shuffle dataset
    dataset = np.concatenate([X_test, y_test], axis = 1)
    # dataset = shuffle(dataset)
    
    X_test = dataset[:,1:25]
    y_test = dataset[:,25:]
    
    # Applying the pre-fitted standard scaler to the test set
    X_test = sc_X.transform(X_test) 
    
   
    # Identifying different athletes IDs based on the filename
    athletes = df.iloc[:, 0].values
    fname = []
    ctr = 0
    reps = 0
    athlete = {}
    
    for ii in range(len(athletes)):        

        fname = athletes[ii]
        # fname = ''.join(ii for ii in fname if not ii.isdigit())
        fname = fname.replace('video', 'Athlete')
        fname = fname.replace('.csv', '')
        fname = fname[0:10]
        
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
                
    return X_test, y_test, athlete

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
    clf6 = load(str(clfs[5]))
    
    return clf1, clf2, clf3, clf4, clf5, clf6

def predictions(clf1, clf2, clf3, X_test, y_test):
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
    # Majority voting and confidence interval estimation
    preds = np.zeros((len(X_test),3))
    votes = np.zeros(len(X_test))
    mean = np.zeros(len(X_test))
    stddev = np.zeros(len(X_test))
    res = np.zeros(len(X_test))   
    conf = np.zeros(len(X_test))   
    
    for ii in range(len(X_test)):
       preds[ii,0] = y_pred1[ii]  
       preds[ii,1] = y_pred2[ii]
       preds[ii,2] = y_pred3[ii]  
       votes[ii] = np.count_nonzero(preds[ii,:])
             
       # Voting for class 0: Mistake 1 not present
       if votes[ii] < 2:
           temp = 0
       # Voting for class 1: Mistake 1 present
       else:
           temp = 1
           
       # Storing all the probabilities into one array
       probs = np.array([y_pred1_prob[ii,temp], y_pred2_prob[ii,temp], \
                         y_pred3_prob[ii,temp]])   
       
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


def report_gen(res, res_1, res_2, conf, conf_1, conf_2, athlete_db, plt_):
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
    m1 = np.zeros(len(athlete_db.keys()))
    m2 = np.zeros(len(athlete_db.keys()))
    low_conf = np.zeros(len(athlete_db.keys()))
    
    for ii in athlete_db.keys():
        list_keys.append(ii)
        reps = athlete_db.get(str(ii))
        correct[ctr] = np.count_nonzero(res[pos:pos + reps]) 
        print('--------------------------')
        string = 'The athlete {} carried out {} correct reps out of {}'.\
            format(str(ii),int(correct[ctr]),int(reps))
        print(string)
        
        m1[ctr] = np.count_nonzero(res_1[pos:pos + reps]) 
        m2[ctr] = np.count_nonzero(res_2[pos:pos + reps]) 
        print('--------------------------')
        string = 'Of the remaining, {} were of category "mistake 1", and {} were of category "mistake 2"'.\
            format(int(m1[ctr]),int(m2[ctr]))
        print(string)
        incorrect[ctr] = reps - correct[ctr]
        low_conf[ctr] = np.count_nonzero(conf[pos:pos + reps] < 0.70) 
        low_conf[ctr] = low_conf[ctr] + \
            np.count_nonzero(conf_1[pos:pos + reps] < 0.70) 
        low_conf[ctr] = low_conf[ctr] + \
            np.count_nonzero(conf_2[pos:pos + reps] < 0.70) 
        string = 'Out of {}, {} labels were classified with low confidence'.\
            format(int(reps*3), int(low_conf[ctr]))
        print(string)
        print('--------------------------')
        
        if plt_ == 1:
            plt.figure(ctr+1, dpi = 200)
            plt.pie([correct[ctr],m1[ctr], m2[ctr]], \
                    labels=['Correct Reps', 'Mistake 1', 'Mistake 2'], \
                        startangle=90,autopct='%1d%%', shadow = True)
            plt.title('{:s}'.format(str(ii)))
            ctr = ctr + 1
            pos = pos + reps

        
######################################################################
# Using an ensemble of the top 3 models to predict probability of both mistakes
# and therefore, provide a confidence interval

######################################################################
# SECTION 1: LOAD SCALER FROM TRAINING PHASE

# Specify relative path of all data: training and test
data_path = 'Data/'

sc_X = import_scaler(data_path)


######################################################################
# SECTION 2: PRE-PROCESS THE TEST DATA

# Specify relative path of training raw data (spit into reps)
IN_PATH = 'Test_keypoints/'

# Specify relative path of where the store the extracted angles/vectors
OUT_PATH = 'Test_Vectors/'

process_test_data(IN_PATH, OUT_PATH)


######################################################################
# SECTION 3: LOAD TEST DATA (PRE-SPLIT INTO REPS & IN .csv)

# Loading all the test data and athlete information
(X_test,y_test,athlete_db) = load_test_data(data_path,sc_X)


######################################################################
# SECTION 4: IMPORT CLASSIFIERS

# Specify relative path of all trained classifier models
clf_path = 'Trained_models/'
# Caution with alphabetical order
(clf2, clf5, clf1, clf4, clf6, clf3) = load_all_clf(clf_path)


######################################################################
# SECTION 5: MAKE PREDICTIONS

# Making predictions for each for the samples in X_test, irrespective 
# ... of the athlete in question

# Mistake 1
X_test_1 = X_test[:,0:12]
y_test_1 = y_test[:,1]

(res_1,conf_1) = predictions(clf1,clf2,clf3,X_test_1,y_test_1)

# Mistake 1
X_test_2 = X_test[:,12:]
y_test_2 = y_test[:,2]

(res_2,conf_2) = predictions(clf4,clf5,clf6,X_test_2,y_test_2)


# Combining the results to assess if the rep is correct or incorrect
# Initializations
preds = np.zeros((len(X_test),2))
votes = np.zeros(len(X_test))
mean = np.zeros(len(X_test))
stddev = np.zeros(len(X_test))
res = np.zeros(len(X_test))   
conf = np.zeros(len(X_test)) 


for ii in range(len(X_test)):
       
    # For model 1
    preds[ii,0] = res_1[ii]
    preds[ii,1] = res_2[ii]
    votes[ii] = np.count_nonzero(preds[ii,:])  
    
    # Voting for class 0: Mistake committed
    if votes[ii] >= 1:

        conf[ii] = (conf_1[ii] + conf_2[ii])/2
        temp = 0
        
    # Voting for class 1: No mistake committed
    elif votes[ii] == 0: 
        
        conf[ii] = (conf_1[ii] + conf_2[ii])/2
        temp = 1
        
    res[ii] = temp 

print('--------------------------')

# Ensuring data types are correct
y_test = np.array(y_test, dtype = np.float)

 
# Calculating scores: Accuracy
print('Accuracy Score of correct reps is: {:.2f}%'.\
      format(accuracy_score(y_test[:,0],res)*100))
print('Accuracy Score of mistake 1 is: {:.2f}%'.\
      format(accuracy_score(y_test[:,1],res_1)*100))    
print('Accuracy Score of mistake 2 is: {:.2f}%'.\
      format(accuracy_score(y_test[:,2],res_2)*100)) 
      
acc_corr = accuracy_score(y_test[:,0], res)*100
acc_corr = np.array(acc_corr)

# F1 score
print('F1 score of correct reps is: {:.2f}%'.\
      format(f1_score(y_test[:,0],res, average = 'weighted')*100))
print('F1 score of mistake 1 is: {:.2f}%'.\
      format(f1_score(y_test[:,1],res_1, average = 'weighted')*100))    
print('F1 score of mistake 2 is: {:.2f}%'.\
      format(f1_score(y_test[:,2],res_2, average = 'weighted')*100))  
    
f1_corr = f1_score(y_test[:,0], res, average = 'weighted')*100
f1_corr = np.array(f1_corr)

# Log loss
lloss_corr = BinaryCrossentropy(y_test[:,0], conf, res)
print('The log-loss of correct reps is: {:2f}'.\
      format(lloss_corr)) 
print('The log-loss of mistake 1 is: {:2f}'.\
      format(BinaryCrossentropy(y_test[:,1], conf_1, res_1)))
print('The log-loss of mistake 2 is: {:2f}'.\
      format(BinaryCrossentropy(y_test[:,2], conf_2, res_2)))

lloss_corr = np.array(lloss_corr)

print('--------------------------')


######################################################################
# SECTION 6: GENERATE PERFORMANCE REPORT FOR 3 LABELS

# As per the predictions made, plot the performance of different athletes
# Print results in the form of a pie chart?
plt_ = 0

report_gen(res, res_1, res_2, conf, conf_1, conf_2, athlete_db, plt_)
