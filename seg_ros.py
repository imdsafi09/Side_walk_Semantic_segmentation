import sys
import math
from ackermann_msgs.msg import AckermannDriveStamped
#sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages') # in order to import cv2 under python3
#print("pronblem----------------------")
import cv2
import helpers
from helpers import *
#sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages') # append back in order to import rospy
import numpy as np
#import matplotlib.pyplot as plt
import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge, CvBridgeError
from cv_bridge.boost.cv_bridge_boost import getCvType
import time
import random
from random import *
from sensor_msgs.msg import Image
import ros_numpy
from geometry_msgs.msg import Point, PoseStamped,Twist,Vector3
#from PIL import Image, ImageEnhance
#import tensorflow as tf
#import tensorflow.keras
#from tensorflow.keras import backend as K


class Sidewalk():
    def __init__(self):
        #self._pub = rospy.Publisher('result', PointCloud2, queue_size=1)
        self.count = 0
        self.rgb_map = 0.1
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.model_pretrained = tf.keras.models.load_model('model/model_0.00003.h5',custom_objects={'masked_loss': self.masked_loss})
        #self.image_pub = rospy.Publisher("image_topic_2",Image)
        #self.model_pretrained._make_predict_function()
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw",Image,self.callback)
        self.pub = rospy.Publisher('/seg_cmd_vel', Twist, queue_size=20)

    def masked_loss(self,y_true, y_pred):
        gt_validity_mask = tf.cast(tf.greater_equal(y_true[:, :, :, 0], 0), dtype=tf.float32)
        y_true = K.abs(y_true)
        raw_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        masked = gt_validity_mask * raw_loss
        return tf.reduce_mean(masked)

    def callback(self, msg):
        self.new_frame_time = time.time()
        try:
            cv_image2 = self.bridge.imgmsg_to_cv2(msg)
        except CvBridgeError as e:
            print(e)
        #K.clear_session()
        #print("all")
        #cv_image2 = cv2.resize(cv_image2, (256, 256), interpolation=cv2.INTER_LINEAR)
        cv_image2 = cv2.resize(cv_image2, (224, 224), interpolation=cv2.INTER_LINEAR)
        m_rgb = cv2.cvtColor(cv_image2, cv2.COLOR_BGR2RGB)
        #cv_image_frame = cv_image2.astype(np.float32)/255.
        cv_image = np.flip(cv_image2,axis=2).astype(np.float32)/255.

        ret = self.model_pretrained.predict(np.expand_dims(cv_image, axis=0))[0]
        #ret = run_predict(self.model_pretrained, np.expand_dims(cv_image, axis=0), deeplab= False)[0]
        #print("all2")
        #ret =  ret*255.
        #ret = ret.astype(np.uint8)

        ret_amax = np.argmax(ret,axis=2)*255.
        #ret_amax3 = np.zeros_like(cv_image2)
        #ret_amax3[:,:,0] = ret_amax
        #ret_amax3[:,:,1] = ret_amax
        #ret_amax3[:,:,2] = ret_amax

        #print(np.shape(ret_amax))
        #print(type(ret_amax))
        #add_img = cv2.addWeighted(cv_image_frame, 0.4,ret_amax, 0.1,0)
        #cv2.imshow('pred',ret_amax)
        cv2.imshow('ori',m_rgb)
        #cv2.waitKey(1)
        fps = 1/(self.new_frame_time-self.prev_frame_time)
        self.prev_frame_time = self.new_frame_time
        print(fps)

        print(ret_amax.shape)
        height, width = ret_amax.shape
        m = cv2.moments(ret_amax, False)
        try:
            x, y = m['m10']/m['m00'], m['m01']/m['m00']
        except ZeroDivisionError:
            x, y = width/2, height/2
        cv2.circle(ret_amax,(int(x), int(y)), 2,(0,255,0),3)

        cv2.imshow('pred',ret_amax)
        #cv2.imshow("Image window", cv2.resize(final, (int(width_final/2), int(height_final/2))))
        cv2.waitKey(1)

        angle_error = ((width/2) - x)*0.1
        ang = -2.* angle_error
        ang2 = (math.atan2((height-y), ((width/2)-x))*180/3.1416) - 90



        if ang>40:
            ang = 40
        elif ang<-40:
            ang = -40
        else:
            ang=ang

        oldMax = 40
        oldMin = -40
        newMax = 1
        newMin = -1
        #ang = ang*0.0174533
        oldRange = oldMax - oldMin
        if (oldRange == 0):
            new_ang = ang
        else:
            newRange = newMax - newMin
            new_ang =  (((ang-oldMin)*newRange)/oldRange) + newMin

        print(ang, ang2, new_ang)
        #ang_sent = Float32()
        #ang_sent.data = -new_ang
        vel_msg = Twist()
        vel_msg.linear.x = 0.3 #0.5
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = -new_ang #-(ang*0.0174533)
        self.pub.publish(vel_msg)








def main(args):
  tensor = Sidewalk()
  rospy.init_node('side_walk_seg_nuc', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
