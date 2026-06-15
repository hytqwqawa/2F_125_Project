#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


class LineFollower:

    def __init__(self):
        self.bridge = CvBridge()

        # ==========================
        # 动态读取话题参数
        # ==========================
        binary_topic = rospy.get_param("~binary_topic", "/car1/binary_image")
        cmd_topic = rospy.get_param("~cmd_topic", "/car1/cmd_vel")

        # ==========================
        # 动态读取算法参数
        # ==========================
        self.forward_speed = rospy.get_param("~forward_speed", 0.3)
        self.kp = rospy.get_param("~kp", 0.003)
        self.max_angular = rospy.get_param("~max_angular", 1.2)
        self.min_black_pixels = rospy.get_param("~min_black_pixels", 70)

        rospy.Subscriber(
            binary_topic,
            Image,
            self.image_callback,
            queue_size=1
        )

        self.cmd_pub = rospy.Publisher(
            cmd_topic,
            Twist,
            queue_size=1
        )

        rospy.loginfo("Line follower started on topic: %s", binary_topic)

    def image_callback(self, msg):

        try:
            binary = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="mono8"
            )
        except Exception as e:
            rospy.logerr(str(e))
            return

        height, width = binary.shape
        roi_y_start = int(height * 0.70)
        roi = binary[roi_y_start:height, :]
        roi_inv = 255 - roi

        M = cv2.moments(roi_inv)
        cmd = Twist()

        if M["m00"] < self.min_black_pixels:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            rospy.logerr("Line lost, stopping node.")
            rospy.signal_shutdown("Line lost")
            return

        cx = int(M["m10"] / M["m00"])
        image_center = width / 2.0
        error = cx - image_center
        angular = -self.kp * error

        angular = max(
            -self.max_angular,
            min(self.max_angular, angular)
        )

        cmd.linear.x = self.forward_speed
        cmd.angular.z = angular

        self.cmd_pub.publish(cmd)

        rospy.loginfo_throttle(
            0.5,
            "cx=%d error=%.1f angular=%.2f" % (cx, error, angular)
        )


if __name__ == "__main__":
    rospy.init_node("black_follow")
    LineFollower()
    rospy.spin()