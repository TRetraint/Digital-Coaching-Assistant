import pandas as pd
import numpy as np

START_SQUATS_UPPER_ANGLE_THRESHOLD = 40
END_PULL_UPS_UPPER_ANGLE_THRESHOLD = 130
TIME_FRAME_LIST = 20

reps_position = []
count_reps = 0
in_reps = 0
precedent_pos = 0

df_reps = pd.DataFrame(columns=['x_Nose','y_Nose','x_Neck','y_Neck','x_RShoulder','y_RShoulder','x_RElbow',
'y_RElbow','x_RWrist','y_RWrist','x_LShoulder','y_LShoulder','x_LElbow','y_LElbow','x_LWrist','y_LWrist',
'x_RHip','y_RHip','x_RKnee','y_RKnee','x_RAnkle','y_RAnkle','x_LHip','y_LHip','x_LKnee','y_LKnee','x_LAnkle','y_LAnkle',
'x_REye','y_REye','x_LEye','y_LEye','x_REar','y_REar','x_LEar','y_LEar','Right_Up_Angle','Left_Up_Angle','Right_Low_Angle','Left_Low_Angle'])

def start_reps_squats(right_low_angle,left_low_angle):
    if right_low_angle < START_SQUATS_UPPER_ANGLE_THRESHOLD and left_low_angle < START_SQUATS_UPPER_ANGLE_THRESHOLD:
        return 1
    else:
        return 0

def count_squats_rep(pos_list,right_low_angle,left_low_angle):
    if right_low_angle > 80  and left_low_angle > 80 and mean_list(pos_list) >= 0.2:
        return [] ,1
    else:
        return pos_list,0

def mean_list(pos_list):
    if len(pos_list) < TIME_FRAME_LIST :
        return 0
    else:
        return sum(pos_list[-TIME_FRAME_LIST:])/TIME_FRAME_LIST

df_human = pd.read_csv('./keypoints/squats_video001.csv')
del df_human['Unnamed: 0']

for k in range(len(df_human[:400])):
    print(k)
    val_start_reps = start_reps_squats(df_human['Right_Low_Angle'][k],df_human['Left_Low_Angle'][k])
    reps_position.append(val_start_reps)
    reps_position, val_count = count_squats_rep(reps_position,df_human['Right_Low_Angle'][k], df_human['Left_Low_Angle'][k])
    if val_count:
        count_reps = count_reps + 1
    if val_start_reps:
        in_reps = 1
    if in_reps:
        df_reps = df_reps.append(df_human.iloc[k])
        if precedent_pos == 0 and val_start_reps == 1:
            if count_reps == 0:
                pass
            else:
                if len(df_reps) <= 25:
                    df_reps = pd.DataFrame(columns=['x_Nose','y_Nose','x_Neck','y_Neck','x_RShoulder','y_RShoulder','x_RElbow',
'y_RElbow','x_RWrist','y_RWrist','x_LShoulder','y_LShoulder','x_LElbow','y_LElbow','x_LWrist','y_LWrist',
'x_RHip','y_RHip','x_RKnee','y_RKnee','x_RAnkle','y_RAnkle','x_LHip','y_LHip','x_LKnee','y_LKnee','x_LAnkle','y_LAnkle',
'x_REye','y_REye','x_LEye','y_LEye','x_REar','y_REar','x_LEar','y_LEar','Right_Up_Angle','Left_Up_Angle','Right_Low_Angle','Left_Low_Angle'])
                else:
                    print(df_reps)
                    df_reps = pd.DataFrame(columns=['x_Nose','y_Nose','x_Neck','y_Neck','x_RShoulder','y_RShoulder','x_RElbow',
    'y_RElbow','x_RWrist','y_RWrist','x_LShoulder','y_LShoulder','x_LElbow','y_LElbow','x_LWrist','y_LWrist',
    'x_RHip','y_RHip','x_RKnee','y_RKnee','x_RAnkle','y_RAnkle','x_LHip','y_LHip','x_LKnee','y_LKnee','x_LAnkle','y_LAnkle',
    'x_REye','y_REye','x_LEye','y_LEye','x_REar','y_REar','x_LEar','y_LEar','Right_Up_Angle','Left_Up_Angle','Right_Low_Angle','Left_Low_Angle'])
    precedent_pos = val_start_reps