#!/usr/bin/env python3
# coding=utf-8

import rospy
import math
import sys
from geometry_msgs.msg import Twist, PoseStamped
from tf.transformations import euler_from_quaternion


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class FixedPointArrivalNode(object):
    def __init__(self):
        rospy.init_node('fixed_point_arrival', anonymous=True)

        self.pose_topic = rospy.get_param('~pose_topic', '/vrpn_client_node/car1/pose')
        self.cmd_vel_topic = rospy.get_param('~cmd_vel_topic', '/car1/cmd_vel')

        self.goal_x = rospy.get_param('~goal_x', 3.5)
        self.goal_y = rospy.get_param('~goal_y', -1.8)
        self.pos_tolerance = rospy.get_param('~pos_tolerance', 0.05)
        self.yaw_tolerance = rospy.get_param('~yaw_tolerance', 0.05)
        self.max_lin = rospy.get_param('~max_lin', 0.20)
        self.max_lat = rospy.get_param('~max_lat', 0.20)
        self.max_ang = rospy.get_param('~max_ang', 0.35)
        self.kp_linear = rospy.get_param('~kp_linear', 0.8)
        self.kp_lateral = rospy.get_param('~kp_lateral', 0.8)
        self.kp_yaw = rospy.get_param('~kp_yaw', 1.2)

        self.current_pose = None
        self.target_yaw = None
        self.reached = False

        rospy.loginfo('FixedPointArrivalNode: pose_topic=%s cmd_vel_topic=%s goal=(%.2f, %.2f)',
                      self.pose_topic, self.cmd_vel_topic, self.goal_x, self.goal_y)

        self.pose_sub = rospy.Subscriber(self.pose_topic, PoseStamped, self.pose_callback, queue_size=1)
        self.vel_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=10)

    def pose_callback(self, msg):
        self.current_pose = msg.pose

        if self.target_yaw is None:
            q = msg.pose.orientation
            (_, _, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])
            self.target_yaw = yaw
            rospy.loginfo('FixedPointArrivalNode: initial yaw set from mocap = %.3f rad', self.target_yaw)

    def compute_cmd(self):
        if self.current_pose is None or self.target_yaw is None:
            return None

        pos = self.current_pose.position
        q = self.current_pose.orientation
        (_, _, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])

        dx = self.goal_x - pos.x
        dy = self.goal_y - pos.y

        cmd = Twist()

        distance = math.hypot(dx, dy)
        if distance < self.pos_tolerance:
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            self.reached = True
        else:
            self.reached = False
            speed = self.kp_linear * distance
            max_speed = min(self.max_lin, self.max_lat)
            speed = min(speed, max_speed)
            vx_world = speed * dx / distance
            vy_world = speed * dy / distance

            cmd.linear.x = math.cos(yaw) * vx_world + math.sin(yaw) * vy_world
            cmd.linear.y = -math.sin(yaw) * vx_world + math.cos(yaw) * vy_world

        yaw_err = normalize_angle(self.target_yaw - yaw)
        if abs(yaw_err) > self.yaw_tolerance:
            cmd.angular.z = max(min(self.kp_yaw * yaw_err, self.max_ang), -self.max_ang)
            self.reached = False
        else:
            cmd.angular.z = 0.0

        return cmd

    def spin(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            cmd = self.compute_cmd()
            if cmd is not None:
                self.vel_pub.publish(cmd)
                if self.reached:
                    rospy.loginfo('FixedPointArrivalNode: goal reached, stopping and exiting.')
                    break
            rate.sleep()

        self.stop_robot()
        sys.exit(0)

    def stop_robot(self):
        stop = Twist()
        self.vel_pub.publish(stop)


if __name__ == '__main__':
    try:
        node = FixedPointArrivalNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
