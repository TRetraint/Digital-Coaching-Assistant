START_PULL_UPS_UPPER_ANGLE_THRESHOLD = 40
END_PULL_UPS_UPPER_ANGLE_THRESHOLD = 130
TIME_FRAME_LIST = 20

def start_reps_pull_ups(right_upper_angle,left_upper_angle,y_RWrist,y_LWrist,y_RElbow,y_LElbow):
    if right_upper_angle < START_PULL_UPS_UPPER_ANGLE_THRESHOLD and left_upper_angle < START_PULL_UPS_UPPER_ANGLE_THRESHOLD and y_RWrist < y_RElbow and y_LWrist < y_LElbow:
        return 1
    else:
        return 0

def end_reps_pull_ups(right_upper_angle,left_upper_angle):
    if right_upper_angle > END_PULL_UPS_UPPER_ANGLE_THRESHOLD and left_upper_angle > END_PULL_UPS_UPPER_ANGLE_THRESHOLD:
        return 1
    else:
        return 0


def mean_list(pos_list):
    if len(pos_list) < TIME_FRAME_LIST :
        return 0
    else:
        return sum(pos_list[-TIME_FRAME_LIST:])/TIME_FRAME_LIST
        

def count_pull_ups_rep(pos_list,right_upper_angle,left_upper_angle):
    if right_upper_angle > 80  and left_upper_angle > 80 and mean_list(pos_list) >= 0.25:
        return [] ,1
    else:
        return pos_list,0
