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

df = pd.DataFrame(columns=['x_Nose','y_Nose','x_Neck','y_Neck','x_RShoulder','y_RShoulder','x_RElbow',
'y_RElbow','x_RWrist','y_RWrist','x_LShoulder','y_LShoulder','x_LElbow','y_LElbow','x_LWrist','y_LWrist',
'x_RHip','y_RHip','x_RKnee','y_RKnee','x_RAnkle','y_RAnkle','x_LHip','y_LHip','x_LKnee','y_LKnee','x_LAnkle','y_LAnkle',
'x_REye','y_REye','x_LEye','y_LEye','x_REar','y_REar','x_LEar','y_LEar','Right_Up_Angle','Left_Up_Angle','Right_Low_Angle','Left_Low_Angle'])

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
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('video8Processed.mp4', 0x7634706d,30.0,(int(w),int(h)))
    while True:
        liste = []
        ret_val, image = cam.read()
        logger.debug('image process+')
        if ret_val == False:
            df.to_csv("./pull_up_data/keypoints/" + os.path.basename(os.path.normpath(os.path.splitext(args.camera)[0]))+'.csv')
            break
        
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        logger.debug('postprocess+')
        image, liste = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        liste.append(joint_extractor.Right_Up_Angle(liste))
        liste.append(joint_extractor.Left_Up_Angle(liste))
        liste.append(joint_extractor.Right_Low_Angle(liste))
        liste.append(joint_extractor.Left_Low_Angle(liste))
        df = df.append(pd.Series(liste,index=['x_Nose','y_Nose','x_Neck','y_Neck','x_RShoulder','y_RShoulder','x_RElbow',
'y_RElbow','x_RWrist','y_RWrist','x_LShoulder','y_LShoulder','x_LElbow','y_LElbow','x_LWrist','y_LWrist',
'x_RHip','y_RHip','x_RKnee','y_RKnee','x_RAnkle','y_RAnkle','x_LHip','y_LHip','x_LKnee','y_LKnee','x_LAnkle','y_LAnkle',
'x_REye','y_REye','x_LEye','y_LEye','x_REar','y_REar','x_LEar','y_LEar','Right_Up_Angle','Left_Up_Angle','Right_Low_Angle','Left_Low_Angle']), ignore_index= True)
        logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Start Reps"
        text2 = "End Reps"
        textsize = cv2.getTextSize(text, font, 0.5, 1)[0]
        textsize2 = cv2.getTextSize(text2, font, 0.5, 1)[0]
        scale = 0.05 
        fontScale = min(image.shape[1],image.shape[0])/(25/scale)  
        textX = (image.shape[1] - textsize[0]) / 2
        textY = (image.shape[0] + textsize[1]) - 50
        text2X = (image.shape[1] - textsize2[0]) / 2
        text2Y = (image.shape[0] + textsize2[1]) - 50
        reps_position.append(reps_counter.start_reps_pull_ups(liste[36], liste[37]))
        if reps_counter.start_reps_pull_ups(liste[36], liste[37]):
            cv2.putText(image, text, (int(textX), int(textY) ), font, fontScale, (0, 255, 0), 2)
        if reps_counter.end_reps_pull_ups(liste[36], liste[37]):
            cv2.putText(image, text2, (int(text2X), int(text2Y) ), font, fontScale, (0, 255, 0), 2)
        cv2.imshow('Pose Detection', image)
        fps_time = time.time()
        out.write(image)
        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')
    cv2.destroyAllWindows()


