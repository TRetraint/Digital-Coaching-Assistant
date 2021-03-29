import pandas as pd
import numpy as np

START_PULL_UPS_UPPER_ANGLE_THRESHOLD = 40
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

def start_reps_pull_ups(right_upper_angle,left_upper_angle,y_RWrist,y_LWrist,y_RElbow,y_LElbow):
    if right_upper_angle < START_PULL_UPS_UPPER_ANGLE_THRESHOLD and left_upper_angle < START_PULL_UPS_UPPER_ANGLE_THRESHOLD and y_RWrist < y_RElbow and y_LWrist < y_LElbow:
        return 1
    else:
        return 0

def count_pull_ups_rep(pos_list,right_upper_angle,left_upper_angle):
    if right_upper_angle > 80  and left_upper_angle > 80 and mean_list(pos_list) >= 0.2:
        return [] ,1
    else:
        return pos_list,0

def mean_list(pos_list):
    if len(pos_list) < TIME_FRAME_LIST :
        return 0
    else:
        return sum(pos_list[-TIME_FRAME_LIST:])/TIME_FRAME_LIST


df_human = pd.read_csv('./keypoints/IMG_6606human_1.csv')
del df_human['Unnamed: 0']

for k in range(len(df_human[:800])):
    print(k)
    val_start_reps = start_reps_pull_ups(df_human['Right_Up_Angle'][k],df_human['Left_Up_Angle'][k],df_human['y_RWrist'][k],df_human['y_LWrist'][k],df_human['y_RElbow'][k],df_human['y_LElbow'][k])
    reps_position.append(val_start_reps)
    reps_position, val_count = count_pull_ups_rep(reps_position,df_human['Right_Up_Angle'][k], df_human['Left_Up_Angle'][k])
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
                if len(df_reps) <= 30:
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
