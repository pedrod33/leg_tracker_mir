#!/usr/bin/python3

import rospy
from leg_tracker.msg import Person, PersonArray, Leg, LegArray
from visualization_msgs.msg import Marker
import numpy as np
from geometry_msgs.msg import PointStamped
import tf
from leg_tracker.srv import Classification_TF, Classification_TFResponse
import os
import pandas as pd

import tensorflow as tflow
from tensorflow import keras

        

def tf_classification(req):
    feat = list(req.features)
    feat = np.array(feat)
    if not feat.size%17:
        feat = feat.reshape(int(feat.size/17),17)
        feat = pd.DataFrame(feat)
        feat.replace([np.inf, -np.inf], np.nan, inplace=True)
        feat = feat.dropna()
        feat = feat.drop(feat.columns[6], axis=1)

        #features
        if(feat.shape[1]==16):
            res = ml_model.predict(feat)
            res = [1 if res[i][0] > 0.5 else 0 for i in range(len(res))]
            return Classification_TFResponse(res)
    return Classification_TFResponse([])

if __name__ == '__main__':
    ml_model_folder = rospy.get_param("/detect_leg_clusters/ml_model_file","NN_190.h5")
    ml_model = tflow.keras.models.load_model(ml_model_folder)

    rospy.init_node('detect_leg_clusters_tf', anonymous=True)
    serv = rospy.Service('/classification_tf', Classification_TF, tf_classification)
    rospy.spin()
    
