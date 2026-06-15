#!/usr/bin/env python3
# coding=utf-8

import rospy
import math
import tf
import sys

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist

class VehicleState:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.speed = 0.0
        self.last_time = 0.0
        self.ready = False

    def update(self, msg):
        current_time = msg.header.stamp.to_sec()
        
        # 依据两次位置差与时间差计算速度
        if self.ready and current_time > self.last_time:
            dt = current_time - self.last_time
            dx = msg.pose.position.x - self.x
            dy = msg.pose.position.y - self.y
            self.speed = math.sqrt(dx * dx + dy * dy) / dt

        self.x = msg.pose.position.x
        self.y = msg.pose.position.y
        self.last_time = current_time

        quat = (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        )

        _, _, self.yaw = tf.transformations.euler_from_quaternion(quat)

        self.ready = True


class FormationFollower:
    def __init__(self):
        rospy.init_node("formation_follow")

        # ==================================================
        # 参数
        # ==================================================
        self.leader_name = rospy.get_param("~leader_name", "car3")
        self.follower1_name = rospy.get_param("~follower1_name", "car1")
        self.follower2_name = rospy.get_param("~follower2_name", "car2")

        self.L = rospy.get_param("~L", 0.6)

        alpha_deg = rospy.get_param("~alpha_deg", 60.0)
        self.alpha = math.radians(alpha_deg)

        self.k2 = rospy.get_param("~k2", 1.0)
        self.k3 = rospy.get_param("~k3", 1.0)

        self.kp_yaw = rospy.get_param("~kp_yaw", 3.0)

        self.max_speed = rospy.get_param("~max_speed", 0.6)
        self.max_angular = rospy.get_param("~max_angular", 1.5)

        # ==================================================
        # 场地安全边界参数
        # ==================================================
        self.bound_x_min = rospy.get_param("~bound_x_min", -2.3)
        self.bound_x_max = rospy.get_param("~bound_x_max",  3.6)
        self.bound_y_min = rospy.get_param("~bound_y_min", -1.9)
        self.bound_y_max = rospy.get_param("~bound_y_max",  1.7)

        # ==================================================
        # 车辆状态
        # ==================================================
        self.leader = VehicleState()
        self.follower1 = VehicleState()
        self.follower2 = VehicleState()

        # ==================================================
        # 动捕订阅
        # ==================================================
        rospy.Subscriber(
            "/vrpn_client_node/{}/pose".format(self.leader_name),
            PoseStamped,
            self.leader_callback
        )

        rospy.Subscriber(
            "/vrpn_client_node/{}/pose".format(self.follower1_name),
            PoseStamped,
            self.follower1_callback
        )

        rospy.Subscriber(
            "/vrpn_client_node/{}/pose".format(self.follower2_name),
            PoseStamped,
            self.follower2_callback
        )

        # ==================================================
        # 控制发布
        # ==================================================
        self.f1_pub = rospy.Publisher(
            "/{}/cmd_vel".format(self.follower1_name),
            Twist,
            queue_size=10
        )

        self.f2_pub = rospy.Publisher(
            "/{}/cmd_vel".format(self.follower2_name),
            Twist,
            queue_size=10
        )

        rospy.loginfo("Formation follower started")
        rospy.loginfo("Leader : {}".format(self.leader_name))
        rospy.loginfo("Follower1 : {}".format(self.follower1_name))
        rospy.loginfo("Follower2 : {}".format(self.follower2_name))
        rospy.loginfo("Bounds: X[{}, {}], Y[{}, {}]".format(
            self.bound_x_min, self.bound_x_max, self.bound_y_min, self.bound_y_max
        ))

    # ==================================================
    # 回调
    # ==================================================
    def leader_callback(self, msg):
        self.leader.update(msg)

    def follower1_callback(self, msg):
        self.follower1.update(msg)

    def follower2_callback(self, msg):
        self.follower2.update(msg)

    # ==================================================
    # 边界检查
    # ==================================================
    def check_boundaries(self):
        vehicles = [
            (self.leader_name, self.leader),
            (self.follower1_name, self.follower1),
            (self.follower2_name, self.follower2)
        ]

        for name, state in vehicles:
            if not state.ready:
                continue

            if (state.x < self.bound_x_min or state.x > self.bound_x_max or
                state.y < self.bound_y_min or state.y > self.bound_y_max):
                
                rospy.logwarn_throttle(
                    1.0,
                    "SAFETY TRIGGERED: Vehicle {} is out of bounds! "
                    "Pos(x: {:.2f}, y: {:.2f}). Waiting to return...".format(
                        name, state.x, state.y
                    )
                )
                return True
                
        return False

    # ==================================================
    # 角度归一化
    # ==================================================
    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    # ==================================================
    # ux uy -> cmd_vel
    # ==================================================
    def velocity_to_cmd(self, ux, uy, current_yaw):
        desired_yaw = math.atan2(uy, ux)

        speed = math.sqrt(
            ux * ux +
            uy * uy
        )

        speed = min(speed, self.max_speed)

        yaw_error = self.normalize_angle(
            desired_yaw - current_yaw
        )

        omega = self.kp_yaw * yaw_error

        omega = max(
            -self.max_angular,
            min(self.max_angular, omega)
        )

        cmd = Twist()
        cmd.linear.x = speed
        cmd.angular.z = omega

        return cmd

    # ==================================================
    # follower1 跟踪 leader
    # ==================================================
    def compute_u1(self):
        theta3 = self.leader.yaw

        ux = -self.k2 * (
            self.follower1.x
            - self.leader.x
            + self.L * math.sin(self.alpha) * math.cos(theta3)
            + self.L * math.cos(self.alpha) * math.sin(theta3)
        )

        uy = -self.k2 * (
            self.follower1.y
            - self.leader.y
            + self.L * math.sin(self.alpha) * math.sin(theta3)
            - self.L * math.cos(self.alpha) * math.cos(theta3)
        )

        return ux, uy

    # ==================================================
    # follower2 跟踪 follower1
    # ==================================================
    def compute_u2(self, u1x, u1y):
        theta3 = self.leader.yaw

        ux = (
            u1x
            - self.k3 * (
                self.follower2.x
                - self.follower1.x
                - 2.0 * self.L
                * math.cos(self.alpha)
                * math.sin(theta3)
            )
        )

        uy = (
            u1y
            - self.k3 * (
                self.follower2.y
                - self.follower1.y
                + 2.0 * self.L
                * math.cos(self.alpha)
                * math.cos(theta3)
            )
        )

        return ux, uy

    # ==================================================
    # 停车 (保留函数供其他逻辑扩展调用)
    # ==================================================
    def stop_all(self):
        cmd = Twist()
        self.f1_pub.publish(cmd)
        self.f2_pub.publish(cmd)

    # ==================================================
    # 主循环
    # ==================================================
    def run(self):
        rate = rospy.Rate(20)

        rospy.loginfo("Waiting mocap...")

        while not rospy.is_shutdown():

            if not (
                self.leader.ready and
                self.follower1.ready and
                self.follower2.ready
            ):
                rate.sleep()
                continue

            # ==================================================
            # 每轮控制前执行安全边界检查
            # ==================================================
            if self.check_boundaries():
                self.stop_all()
                rate.sleep()
                continue

            # ----------------------------------
            # follower1 控制计算
            # ----------------------------------
            u1x, u1y = self.compute_u1()

            cmd1 = self.velocity_to_cmd(
                u1x,
                u1y,
                self.follower1.yaw
            )

            self.f1_pub.publish(cmd1)

            # ----------------------------------
            # follower2 控制计算
            # ----------------------------------
            u2x, u2y = self.compute_u2(
                u1x,
                u1y
            )

            cmd2 = self.velocity_to_cmd(
                u2x,
                u2y,
                self.follower2.yaw
            )

            self.f2_pub.publish(cmd2)

            rospy.loginfo_throttle(
                0.5,
                "u1=(%.2f %.2f) w1=%.2f | u2=(%.2f %.2f) w2=%.2f",
                u1x,
                u1y,
                cmd1.angular.z,
                u2x,
                u2y,
                cmd2.angular.z
            )

            rate.sleep()

if __name__ == "__main__":
    try:
        node = FormationFollower()
        node.run()
    except rospy.ROSInterruptException:
        pass