import pandas as pd
import sys

print(sys.path)


df = pd.DataFrame(columns=['x_Nose','y_Nose','x_Neck','y_Neck','x_RShoulder','y_RShoulder','x_RElbow',
'y_RElbow','x_RWrist','y_RWrist','x_LShoulder','y_LShoulder','x_LElbow','y_LElbow','x_LWrist','y_LWrist',
'x_RHip','y_RHip','x_RKnee','y_RKnee','x_RAnkle','y_RAnkle','x_LHip','y_LHip','x_LKnee','y_LKnee','x_LAnkle','y_LAnkle',
'x_REye','y_REye','x_LEye','y_LEye','x_REar','y_REar','x_LEar','y_LEar'])

df = df.append(pd.Series([204, 292, 204, 302, 176, 306, 158, 244, 146, 198, 232, 296, 248, 244, 260, 200, 182, 420, 182, 506, 188, 580, 222, 420, 226, 508, 224, 580, 196, 286, 210, 286, 186, 286, 220, 284]
, index=['x_Nose','y_Nose','x_Neck','y_Neck','x_RShoulder','y_RShoulder','x_RElbow',
'y_RElbow','x_RWrist','y_RWrist','x_LShoulder','y_LShoulder','x_LElbow','y_LElbow','x_LWrist','y_LWrist',
'x_RHip','y_RHip','x_RKnee','y_RKnee','x_RAnkle','y_RAnkle','x_LHip','y_LHip','x_LKnee','y_LKnee','x_LAnkle','y_LAnkle',
'x_REye','y_REye','x_LEye','y_LEye','x_REar','y_REar','x_LEar','y_LEar']), ignore_index=True)

print(df)
import run_webcam
print(run_webcam.df)