import argparse
import logging
import time

import cv2
import numpy as np
import pandas as pd
import os
import joint_extractor
import reps_counter

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0
reps_position = []
count_reps = 0
precedent_pos = 0

in_reps = 0

df_list = []

df_reps = pd.DataFrame(columns=['Right_Up_Angle','Left_Up_Angle','Right_Low_Angle','Left_Low_Angle'])

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=str, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    
    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    print("RESOLUTION {0}x{1}".format(w,h))
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    w = int(cam.get(3))
    h = int(cam.get(4))
    while True:
        coord = []
        ret_val, image = cam.read()
        logger.debug('image process+')
        if ret_val == False:
            for id_human in range(len(df_list)):
                print('fin')
                df_list[id_human].to_csv("./keypoints/" + os.path.basename(os.path.normpath(os.path.splitext(args.camera)[0]))+'human_'+str(id_human+1)+'.csv')
            break
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        logger.debug('postprocess+')
        image, coord = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        if df_list == []:
            for i in range(len(coord)):
                df_list.append(pd.DataFrame(columns=['x_Nose','y_Nose','x_Neck','y_Neck','x_RShoulder','y_RShoulder','x_RElbow',
'y_RElbow','x_RWrist','y_RWrist','x_LShoulder','y_LShoulder','x_LElbow','y_LElbow','x_LWrist','y_LWrist',
'x_RHip','y_RHip','x_RKnee','y_RKnee','x_RAnkle','y_RAnkle','x_LHip','y_LHip','x_LKnee','y_LKnee','x_LAnkle','y_LAnkle',
'x_REye','y_REye','x_LEye','y_LEye','x_REar','y_REar','x_LEar','y_LEar','Right_Up_Angle','Left_Up_Angle','Right_Low_Angle','Left_Low_Angle']))
        for id_human in range(len(coord)):
            rua = joint_extractor.Right_Up_Angle(coord[id_human])
            lua = joint_extractor.Left_Up_Angle(coord[id_human])
            rla = joint_extractor.Right_Low_Angle(coord[id_human])
            lla = joint_extractor.Left_Low_Angle(coord[id_human])
            coord[id_human].append(rua)
            coord[id_human].append(lua)
            coord[id_human].append(rla)
            coord[id_human].append(lla)
            df_list[id_human] = df_list[id_human].append(pd.Series(coord[id_human],index=['x_Nose','y_Nose','x_Neck','y_Neck','x_RShoulder','y_RShoulder','x_RElbow',
'y_RElbow','x_RWrist','y_RWrist','x_LShoulder','y_LShoulder','x_LElbow','y_LElbow','x_LWrist','y_LWrist',
'x_RHip','y_RHip','x_RKnee','y_RKnee','x_RAnkle','y_RAnkle','x_LHip','y_LHip','x_LKnee','y_LKnee','x_LAnkle','y_LAnkle',
'x_REye','y_REye','x_LEye','y_LEye','x_REar','y_REar','x_LEar','y_LEar','Right_Up_Angle','Left_Up_Angle','Right_Low_Angle','Left_Low_Angle']), ignore_index= True)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = "Start Reps"
            text2 = "Middle Reps"
            text3 = "Count Reps: "+ str(count_reps)
            textsize = cv2.getTextSize(text, font, 0.5, 1)[0]
            textsize2 = cv2.getTextSize(text2, font, 0.5, 1)[0]
            textsize3 = cv2.getTextSize(text2, font, 0.5, 1)[0]
            scale = 0.02
            fontScale = min(image.shape[1],image.shape[0])/(25/scale)  
            textX = (image.shape[1] - textsize[0]) / 2
            textY = (image.shape[0] + textsize[1]) - 50
            text2X = (image.shape[1] - textsize2[0]) / 2
            text2Y = (image.shape[0] + textsize2[1]) - 50
            val_start_reps = reps_counter.start_reps_pull_ups(coord[id_human][36], coord[id_human][37],coord[id_human][9],coord[id_human][15],coord[id_human][7],coord[id_human][13])
            reps_position.append(val_start_reps)
            reps_position, val_count = reps_counter.count_pull_ups_rep(reps_position,coord[id_human][36], coord[id_human][37])
            cv2.putText(image, text3, (int(coord[id_human][20]-10), int(coord[id_human][21]+20)), font, fontScale, (0, 255, 0), 2)
            if val_count:
                count_reps = count_reps + 1
            if val_start_reps:
                cv2.putText(image, text,(int(coord[id_human][20]-10), int(coord[id_human][21]+40)), font, fontScale, (0, 255, 0), 2)
                in_reps = 1
            if in_reps:
                df_reps = df_reps.append(pd.Series([rua, lua, rla, lla],index = ['Right_Up_Angle','Left_Up_Angle','Right_Low_Angle','Left_Low_Angle']), ignore_index= True)
                if precedent_pos == 0 and val_start_reps == 1:
                    if count_reps == 0:
                        pass
                    else:
                        #df_reps.to_csv("./pull_up_data/reps_keypoints/" + os.path.basename(os.path.normpath(os.path.splitext(args.camera)[0]))+'_'+str(count_reps)+'.csv')
                        df_reps = pd.DataFrame(columns=['Right_Up_Angle','Left_Up_Angle','Right_Low_Angle','Left_Low_Angle'])
            if reps_counter.end_reps_pull_ups(coord[id_human][36], coord[id_human][37]):
                cv2.putText(image, text2,(int(coord[id_human][20]-10), int(coord[id_human][21]+40)), font, fontScale, (0, 255, 0), 2)
            precedent_pos = val_start_reps
        logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('Pose Detection', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')
    cv2.destroyAllWindows()