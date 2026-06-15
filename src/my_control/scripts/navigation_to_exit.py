#!/usr/bin/env python3
# coding: utf-8

import rospy
import math
import tf
from geometry_msgs.msg import PoseStamped, Twist

class AckermannPointNavigation:
    def __init__(self):
        rospy.init_node('navigation_to_exit', anonymous=True)

        self.target_x = rospy.get_param("~target_x", 3.5)
        self.target_y = rospy.get_param("~target_y", -1.8)

        # 默认值改为通用占位符，实际值将由 launch 文件注入
        self.pose_topic = rospy.get_param("~pose_topic", "/vrpn_client_node/car1/pose")
        self.cmd_topic = rospy.get_param("~cmd_topic", "/car1/cmd_vel")

        self.bound_x_min = rospy.get_param("~bound_x_min", -2.3)
        self.bound_x_max = rospy.get_param("~bound_x_max",  3.6)
        self.bound_y_min = rospy.get_param("~bound_y_min", -1.9)
        self.bound_y_max = rospy.get_param("~bound_y_max",  1.7)

        self.safety_margin = rospy.get_param("~safety_margin", 0.08)
        self.boundary_predict_time = rospy.get_param("~boundary_predict_time", 0.25)

        self.cruise_speed = rospy.get_param("~cruise_speed", 0.8)
        self.min_speed = rospy.get_param("~min_speed", 0.15)
        self.slow_radius = rospy.get_param("~slow_radius", 0.6)

        self.k_yaw = rospy.get_param("~k_yaw", 2.2)
        self.max_angular = rospy.get_param("~max_angular", 1.6)
        self.distance_threshold = rospy.get_param("~distance_threshold", 0.03)

        self.allow_reverse = rospy.get_param("~allow_reverse", True)
        self.switch_to_reverse_angle = math.radians(110.0)
        self.switch_to_forward_angle = math.radians(70.0)
        self.angular_cmd_is_steering = rospy.get_param("~angular_cmd_is_steering", False)

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.pose_ready = False
        self.direction = 1

        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=10)
        rospy.Subscriber(self.pose_topic, PoseStamped, self.pose_callback)

        self.rate = rospy.Rate(50)
        rospy.on_shutdown(self.stop)

        rospy.loginfo("Ackermann point navigation initialized.")
        rospy.loginfo("Target point: x=%.3f, y=%.3f", self.target_x, self.target_y)
        rospy.loginfo("Listening on: %s", self.pose_topic)
        rospy.loginfo("Publishing to: %s", self.cmd_topic)

    def pose_callback(self, msg):
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y

        quat = (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        )
        _, _, self.current_yaw = tf.transformations.euler_from_quaternion(quat)
        self.pose_ready = True

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def sat(self, value, lower, upper):
        return max(lower, min(upper, value))

    def distance_to_target(self):
        dx = self.target_x - self.current_x
        dy = self.target_y - self.current_y
        return math.hypot(dx, dy)

    def los_angle(self):
        dx = self.target_x - self.current_x
        dy = self.target_y - self.current_y
        return math.atan2(dy, dx)

    def inside_safe_area(self, x, y):
        return (
            self.bound_x_min + self.safety_margin <= x <= self.bound_x_max - self.safety_margin and
            self.bound_y_min + self.safety_margin <= y <= self.bound_y_max - self.safety_margin
        )

    def target_is_valid(self):
        return self.inside_safe_area(self.target_x, self.target_y)

    def current_touch_boundary(self):
        return not self.inside_safe_area(self.current_x, self.current_y)

    def predict_touch_boundary(self, v):
        x_next = self.current_x + v * math.cos(self.current_yaw) * self.boundary_predict_time
        y_next = self.current_y + v * math.sin(self.current_yaw) * self.boundary_predict_time
        return not self.inside_safe_area(x_next, y_next)

    def update_direction(self, los):
        if not self.allow_reverse:
            self.direction = 1
            return

        heading_error_forward = abs(self.normalize_angle(los - self.current_yaw))

        if self.direction == 1:
            if heading_error_forward > self.switch_to_reverse_angle:
                self.direction = -1
        else:
            if heading_error_forward < self.switch_to_forward_angle:
                self.direction = 1

    def motion_heading_error(self, los):
        if self.direction == 1:
            return self.normalize_angle(los - self.current_yaw)
        else:
            return self.normalize_angle(los - self.current_yaw - math.pi)

    def compute_cmd(self):
        dist = self.distance_to_target()
        los = self.los_angle()

        self.update_direction(los)
        heading_error = self.motion_heading_error(los)

        if dist < self.slow_radius:
            v_abs = self.cruise_speed * dist / self.slow_radius
            v_abs = max(self.min_speed, v_abs)
        else:
            v_abs = self.cruise_speed

        turn_slow = max(0.25, math.cos(min(abs(heading_error), math.pi / 2.0)))
        v_abs *= turn_slow

        v = self.direction * v_abs
        omega = self.k_yaw * heading_error
        omega = self.sat(omega, -self.max_angular, self.max_angular)

        if self.angular_cmd_is_steering and self.direction == -1:
            omega = -omega

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = omega

        return cmd, dist, heading_error

    def stop(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0

        for _ in range(3):
            self.cmd_pub.publish(cmd)
            rospy.sleep(0.02)

    def run(self):
        rospy.loginfo("Waiting for pose topic: %s", self.pose_topic)

        try:
            first_msg = rospy.wait_for_message(self.pose_topic, PoseStamped, timeout=5.0)
            self.pose_callback(first_msg)
        except rospy.ROSException:
            rospy.logerr("No pose received. Please check VRPN connection.")
            self.stop()
            return

        rospy.loginfo("Pose received.")

        if not self.target_is_valid():
            rospy.logerr("Target point is outside safe area.")
            self.stop()
            return

        while not rospy.is_shutdown():
            if not self.pose_ready:
                self.stop()
                self.rate.sleep()
                continue

            dist = self.distance_to_target()

            if dist <= self.distance_threshold:
                rospy.loginfo("Target reached. pos=(%.3f, %.3f), dist=%.3f", self.current_x, self.current_y, dist)
                self.stop()
                break

            if self.current_touch_boundary():
                rospy.logwarn("Safety boundary reached. Stop.")
                self.stop()
                break

            cmd, dist, heading_error = self.compute_cmd()

            if self.predict_touch_boundary(cmd.linear.x):
                rospy.logwarn("Predicted safety-boundary violation. Stop.")
                self.stop()
                break

            self.cmd_pub.publish(cmd)
            rospy.loginfo_throttle(0.5, "dist=%.3f, v=%.3f, omega=%.3f", dist, cmd.linear.x, cmd.angular.z)
            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = AckermannPointNavigation()
        node.run()
    except rospy.ROSInterruptException:
        pass