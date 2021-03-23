#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

import os
print(os.path.splitext('./pull_up_data/keypoints/video1_1.csv')[0])
test = os.path.splitext('./pull_up_data/keypoints/video1_1.csv')[0]
print(os.path.basename(os.path.normpath(test)))
