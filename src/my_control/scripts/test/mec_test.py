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


class SquareGotoNode(object):
    def __init__(self):
        rospy.init_node('square_goto_node')

        self.mocap_topic = rospy.get_param('~mocap_topic', '/vrpn_client_node/car6/pose')
        self.cmd_vel_topic = rospy.get_param('~cmd_vel_topic', '/car6/cmd_vel')
        self.pos_tolerance = rospy.get_param('~pos_tolerance', 0.05)
        self.yaw_tolerance = rospy.get_param('~yaw_tolerance', 0.05)
        self.max_lin = rospy.get_param('~max_lin', 0.20)
        self.max_lat = rospy.get_param('~max_lat', 0.20)
        self.max_ang = rospy.get_param('~max_ang', 0.35)
        self.kp_linear = rospy.get_param('~kp_linear', 0.8)
        self.kp_yaw = rospy.get_param('~kp_yaw', 1.2)

        self.waypoints = [
            (1.0, 1.0),
            (1.0, -1.0),
            (-1.0, -1.0),
            (-1.0, 1.0),
        ]
        self.current_goal_index = 0
        self.goal_x, self.goal_y = self.waypoints[self.current_goal_index]
        self.completed = False

        self.current_pose = None
        self.target_yaw = None
        self.reached = False

        rospy.loginfo('SquareGotoNode: mocap_topic=%s cmd_vel_topic=%s square waypoints=%s',
                      self.mocap_topic, self.cmd_vel_topic, self.waypoints)

        self.pose_sub = rospy.Subscriber(self.mocap_topic, PoseStamped, self.pose_callback, queue_size=1)
        self.vel_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=10)

    def pose_callback(self, msg):
        self.current_pose = msg.pose

        if self.target_yaw is None:
            q = msg.pose.orientation
            (_, _, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])
            self.target_yaw = yaw
            rospy.loginfo('SquareGotoNode: initial yaw set from mocap = %.3f rad', self.target_yaw)

    def advance_goal(self):
        self.current_goal_index += 1
        if self.current_goal_index >= len(self.waypoints):
            self.completed = True
            rospy.loginfo('SquareGotoNode: completed full square.')
            return

        self.goal_x, self.goal_y = self.waypoints[self.current_goal_index]
        rospy.loginfo('SquareGotoNode: next waypoint %d = (%.2f, %.2f)',
                      self.current_goal_index + 1, self.goal_x, self.goal_y)

    def compute_cmd(self):
        if self.current_pose is None or self.target_yaw is None:
            return None

        pos = self.current_pose.position
        q = self.current_pose.orientation
        (_, _, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])

        dx = self.goal_x - pos.x
        dy = self.goal_y - pos.y

        distance = math.hypot(dx, dy)
        if distance < self.pos_tolerance:
            rospy.loginfo('SquareGotoNode: reached waypoint %d at (%.2f, %.2f)',
                          self.current_goal_index + 1, self.goal_x, self.goal_y)
            self.advance_goal()
            if self.completed:
                return Twist()
            return None

        cmd = Twist()
        self.reached = False

        # 先计算全局误差向量，统一成世界坐标速度
        speed = self.kp_linear * distance
        max_speed = min(self.max_lin, self.max_lat)
        speed = min(speed, max_speed)
        vx_world = speed * dx / distance
        vy_world = speed * dy / distance

        # 将世界坐标速度旋转到机器人坐标系
        cmd.linear.x = math.cos(yaw) * vx_world + math.sin(yaw) * vy_world
        cmd.linear.y = -math.sin(yaw) * vx_world + math.cos(yaw) * vy_world

        yaw_err = normalize_angle(self.target_yaw - yaw)
        if abs(yaw_err) > self.yaw_tolerance:
            cmd.angular.z = max(min(self.kp_yaw * yaw_err, self.max_ang), -self.max_ang)
        else:
            cmd.angular.z = 0.0

        return cmd

    def spin(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and not self.completed:
            cmd = self.compute_cmd()
            if cmd is not None:
                self.vel_pub.publish(cmd)
            rate.sleep()

        self.stop_robot()
        rospy.loginfo('SquareGotoNode: stopped and exiting.')
        sys.exit(0)

    def stop_robot(self):
        stop = Twist()
        self.vel_pub.publish(stop)


if __name__ == '__main__':
    try:
        node = SquareGotoNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
