#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import math
import tf
from geometry_msgs.msg import Twist, PoseStamped

class AffineFollowerPN:
    def __init__(self):
        rospy.init_node('affine_follower', anonymous=True)
        
        self.car_name = rospy.get_param('~car_name', 'car1')
        self.neighbor_topics = rospy.get_param('~neighbors', [])
        
        if len(self.neighbor_topics) != 3:
            rospy.logerr("[{}] Needs exactly 3 neighbors specified.".format(self.car_name))
            return
            
        self.self_pose_topic = '/vrpn_client_node/{}/pose'.format(self.car_name)
        self.cmd_topic = '/{}/cmd_vel'.format(self.car_name)
        self.pub_cmd = rospy.Publisher(self.cmd_topic, Twist, queue_size=10)
        
        self.cruise_speed = rospy.get_param("~cruise_speed", 0.4)
        self.min_speed = rospy.get_param("~min_speed", 0.05)
        self.slow_radius = rospy.get_param("~slow_radius", 0.6)
        self.k_yaw = rospy.get_param("~k_yaw", 2.2)
        self.max_angular = rospy.get_param("~max_angular", 2.2)
        self.distance_threshold = rospy.get_param("~distance_threshold", 0.16)
        
        self.allow_reverse = rospy.get_param("~allow_reverse", True)
        self.switch_to_reverse_angle = math.radians(110.0)
        self.switch_to_forward_angle = math.radians(70.0)
        self.angular_cmd_is_steering = rospy.get_param("~angular_cmd_is_steering", False)
        self.min_separation = rospy.get_param("~min_separation", 0.2)
        self.real_car_indices = [i for i, topic in enumerate(self.neighbor_topics) if 'vrpn_client_node' in topic]
        self.avoidance_speed_factor = rospy.get_param("~avoidance_speed_factor", 0.2)
        self.avoidance_heading_bias = rospy.get_param("~avoidance_heading_bias", math.radians(35.0))
        
        self.num_cars = rospy.get_param("~num_cars", 4)
        self.start_time = rospy.Time.now()
        rospy.set_param('/formation_status/{}'.format(self.car_name), False)
        
        self.self_pos = None
        self.self_yaw = 0.0
        self.neighbor_pos = [None, None, None]
        self.direction = 1
        
        # --- 新增：读取标称位置参数并在初始化时直接计算静态权重 ---
        self.nominal_self_pos = rospy.get_param("~nominal_self_pos", [0.0, 0.0])
        self.nominal_neighbors_pos = rospy.get_param("~nominal_neighbors_pos", [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        
        N_mat = np.ones((3, 3))
        for i in range(3):
            N_mat[0:2, i] = self.nominal_neighbors_pos[i]
            
        try:
            N_mat_inv = np.linalg.inv(N_mat)
            self_aug = np.array([self.nominal_self_pos[0], self.nominal_self_pos[1], 1.0])
            self.weights = np.dot(N_mat_inv, self_aug)
            rospy.loginfo("[{}] Topology locked from nominal positions! Affine Weights: {}".format(self.car_name, self.weights))
        except np.linalg.LinAlgError:
            rospy.logerr("[{}] Collinear nominal positions! Cannot compute weights.".format(self.car_name))
            self.weights = np.zeros(3)
        # ------------------------------------------------------

        rospy.Subscriber(self.self_pose_topic, PoseStamped, self.self_pose_cb)
        for i, topic in enumerate(self.neighbor_topics):
            rospy.Subscriber(topic, PoseStamped, self.neighbor_pose_cb, callback_args=i)
            
        self.rate = rospy.Timer(rospy.Duration(0.02), self.control_loop)

    def self_pose_cb(self, msg):
        self.self_pos = np.array([msg.pose.position.x, msg.pose.position.y])
        q = msg.pose.orientation
        _, _, self.self_yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
            
    def neighbor_pose_cb(self, msg, idx):
        self.neighbor_pos[idx] = np.array([msg.pose.position.x, msg.pose.position.y])

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def sat(self, value, lower, upper):
        return max(lower, min(upper, value))

    def update_direction(self, los):
        if not self.allow_reverse:
            self.direction = 1
            return
        heading_error_forward = abs(self.normalize_angle(los - self.self_yaw))
        if self.direction == 1:
            if heading_error_forward > self.switch_to_reverse_angle:
                self.direction = -1
        else:
            if heading_error_forward < self.switch_to_forward_angle:
                self.direction = 1

    def motion_heading_error(self, los):
        if self.direction == 1:
            return self.normalize_angle(los - self.self_yaw)
        else:
            return self.normalize_angle(los - self.self_yaw - math.pi)

    def control_loop(self, event):
        if self.self_pos is None or any(p is None for p in self.neighbor_pos):
            return
            
        # 动态目标解算（权重现在是静态常数，实时跟踪邻居当前的真实位置）
        target_pos = np.zeros(2)
        for i in range(3):
            target_pos += self.weights[i] * self.neighbor_pos[i]
            
        dx = target_pos[0] - self.self_pos[0]
        dy = target_pos[1] - self.self_pos[1]
        dist = math.hypot(dx, dy)
        los = math.atan2(dy, dx)
        
        cmd = Twist()

        # 近邻避障
        too_close = False
        avoidance_bias = 0.0
        if self.real_car_indices:
            for i in self.real_car_indices:
                neighbor = self.neighbor_pos[i]
                if neighbor is None:
                    continue
                neighbor_dist = math.hypot(neighbor[0] - self.self_pos[0], neighbor[1] - self.self_pos[1])
                if neighbor_dist < self.min_separation:
                    too_close = True
                    neighbor_angle = math.atan2(neighbor[1] - self.self_pos[1], neighbor[0] - self.self_pos[0])
                    angle_diff = self.normalize_angle(neighbor_angle - self.self_yaw)
                    avoidance_bias = -self.avoidance_heading_bias if angle_diff > 0 else self.avoidance_heading_bias
                    break

        # 阈值判断
        if dist <= self.distance_threshold:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.pub_cmd.publish(cmd)
            
            rospy.set_param('/formation_status/{}'.format(self.car_name), True)
            return
        else:
            rospy.set_param('/formation_status/{}'.format(self.car_name), False)

        # 运动学控制
        self.update_direction(los)
        heading_error = self.motion_heading_error(los)

        if dist < self.slow_radius:
            v_abs = self.cruise_speed * dist / self.slow_radius
            v_abs = max(self.min_speed, v_abs)
        else:
            v_abs = self.cruise_speed

        turn_slow = max(0.25, math.cos(min(abs(heading_error), math.pi / 2.0)))
        v_abs *= turn_slow

        if too_close:
            v_abs *= self.avoidance_speed_factor
            heading_error += avoidance_bias
            heading_error = self.normalize_angle(heading_error)

        v = self.direction * v_abs
        omega = self.k_yaw * heading_error
        omega = self.sat(omega, -self.max_angular, self.max_angular)

        if self.angular_cmd_is_steering and self.direction == -1:
            omega = -omega

        cmd.linear.x = v
        cmd.angular.z = omega
        self.pub_cmd.publish(cmd)

if __name__ == '__main__':
    try:
        AffineFollowerPN()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass