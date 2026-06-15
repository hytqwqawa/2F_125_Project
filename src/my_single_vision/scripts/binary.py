#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

class ImageBinarizer:
    def __init__(self):
        self.bridge = CvBridge()
        
        # 1. 动态读取话题名称私有参数 (提供默认值以防未配置 Launch 文件)
        image_topic = rospy.get_param("~image_topic", "/car1/image_raw")
        binary_topic = rospy.get_param("~binary_topic", "/car1/binary_image")
        
        # 2. 读取二值化阈值参数
        self.threshold_value = rospy.get_param("~threshold", 105)
        
        # 3. 初始化订阅与发布
        self.image_sub = rospy.Subscriber(image_topic, Image, self.callback)
        self.binary_pub = rospy.Publisher(binary_topic, Image, queue_size=1)
        
        rospy.loginfo("Subscribed to: %s", image_topic)
        rospy.loginfo("Publishing to: %s", binary_topic)
        rospy.loginfo("Binary threshold set to %d", self.threshold_value)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, self.threshold_value, 255, cv2.THRESH_BINARY)

        try:
            binary_msg = self.bridge.cv2_to_imgmsg(binary, encoding="mono8")
            binary_msg.header = data.header
            self.binary_pub.publish(binary_msg)
        except CvBridgeError as e:
            rospy.logerr(e)

if __name__ == '__main__':
    # 此处的节点名会被 launch 文件中的 name 属性自动覆盖
    rospy.init_node("image_binarizer_node", anonymous=False)
    node = ImageBinarizer()
    rospy.spin()