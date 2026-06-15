#!/usr/bin/env python3
# coding=utf-8

import rospy
import math
import tf

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist


class CarController:

    def __init__(self, car_name, target_x, target_y, desired_impact_time):

        self.car_name = car_name

        self.target_x = target_x
        self.target_y = target_y

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        self.pose_ready = False
        self.reached = False

        self.cruise_speed = 0.3
        self.accel_duration = 0.2

        self.desired_impact_time = desired_impact_time
        self.k = 3.5

        self.a_gain = 3.0
        self.max_angular = 3.0

        self.distance_threshold = 0.08
        self.sigma_threshold_deg = 0.1

        self.filtered_sigma_d_dot = 0.0
        self.filter_alpha = 0.15

        self.STATE_ACCEL = 1
        self.STATE_GUIDANCE = 2
        self.STATE_LINEAR = 3

        self.state = self.STATE_ACCEL

        self.start_time = None

        pose_topic = "/vrpn_client_node/{}/pose".format(car_name)
        cmd_topic = "/{}/cmd_vel".format(car_name)

        rospy.Subscriber(
            pose_topic,
            PoseStamped,
            self.pose_callback
        )

        self.cmd_pub = rospy.Publisher(
            cmd_topic,
            Twist,
            queue_size=10
        )

        rospy.loginfo(
            "{} controller initialized -> target ({:.2f},{:.2f})".format(
                car_name,
                target_x,
                target_y
            )
        )

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

    def get_line_of_sight_angle(self):

        return math.atan2(
            self.target_y - self.current_y,
            self.target_x - self.current_x
        )

    def stop(self):

        self.cmd_pub.publish(Twist())

    def control_step(self):

        if not self.pose_ready:
            return

        if self.reached:
            return

        if self.start_time is None:
            self.start_time = rospy.Time.now()

        current_time = rospy.Time.now()

        elapsed_time = (
            current_time - self.start_time
        ).to_sec()

        t_go_desired = (
            self.desired_impact_time -
            elapsed_time
        )

        dx = self.target_x - self.current_x
        dy = self.target_y - self.current_y

        R = math.hypot(dx, dy)

        V = self.cruise_speed

        if R < self.distance_threshold:

            self.stop()

            self.reached = True

            rospy.loginfo(
                "{} reached target".format(
                    self.car_name
                )
            )

            return

        # =====================
        # 加速阶段
        # =====================

        if self.state == self.STATE_ACCEL:

            if elapsed_time >= self.accel_duration:

                t_go_desired_init = (
                    self.desired_impact_time -
                    elapsed_time
                )

                if t_go_desired_init < R / V:

                    rospy.logerr(
                        "{} impact time impossible".format(
                            self.car_name
                        )
                    )

                    self.stop()

                    self.reached = True

                    return

                self.state = self.STATE_GUIDANCE

                self.filtered_sigma_d_dot = 0.0

            else:

                linear_speed = (
                    V *
                    elapsed_time /
                    self.accel_duration
                )

                cmd = Twist()

                cmd.linear.x = linear_speed
                cmd.angular.z = 0.0

                self.cmd_pub.publish(cmd)

                return

        # =====================
        # ITCG
        # =====================

        if self.state == self.STATE_GUIDANCE:

            lambda_rad = (
                self.get_line_of_sight_angle()
            )

            psi = self.current_yaw

            sigma = self.normalize_angle(
                psi - lambda_rad
            )

            if R > 0.01:

                dlambda_dt = (
                    -V *
                    math.sin(sigma) /
                    R
                )

            else:

                dlambda_dt = 0.0

            epsilon = (
                V * t_go_desired -
                R
            )

            if (
                R < 1.0
                or
                (
                    R < 1.5
                    and
                    abs(epsilon) < 0.05
                )
            ):

                sigma_d = 0.0

                self.filtered_sigma_d_dot = 0.0

                current_a_gain = (
                    self.a_gain * 1.5
                )

            else:

                denom = (
                    R +
                    self.k * epsilon
                )

                if denom <= 0:

                    X = 1.0

                else:

                    X = R / denom

                X = max(
                    -0.999,
                    min(0.999, X)
                )

                sigma_d_mag = math.acos(X)

                sigma_d = math.copysign(
                    sigma_d_mag,
                    sigma
                )

                R_dot = (
                    -V *
                    math.cos(sigma)
                )

                epsilon_dot = (
                    -V *
                    (
                        1.0 -
                        math.cos(sigma)
                    )
                )

                X_dot = (
                    self.k *
                    (
                        R_dot * epsilon
                        -
                        R * epsilon_dot
                    )
                ) / (denom ** 2)

                sigma_d_dot_mag = (
                    -X_dot /
                    math.sqrt(
                        1.0 - X * X
                    )
                )

                raw_sigma_d_dot = (
                    math.copysign(
                        sigma_d_dot_mag,
                        sigma_d
                    )
                )

                raw_sigma_d_dot = max(
                    -2.0,
                    min(
                        2.0,
                        raw_sigma_d_dot
                    )
                )

                self.filtered_sigma_d_dot = (
                    self.filter_alpha *
                    raw_sigma_d_dot
                    +
                    (
                        1.0 -
                        self.filter_alpha
                    )
                    *
                    self.filtered_sigma_d_dot
                )

                current_a_gain = self.a_gain

            e = self.normalize_angle(
                sigma - sigma_d
            )

            omega = (
                dlambda_dt
                +
                self.filtered_sigma_d_dot
                -
                current_a_gain * e
            )

            omega = max(
                -self.max_angular,
                min(
                    self.max_angular,
                    omega
                )
            )

            if (
                abs(
                    sigma *
                    180.0 /
                    math.pi
                )
                <
                self.sigma_threshold_deg
                and
                R < 0.5
            ):

                self.state = self.STATE_LINEAR

                omega = 0.0

            cmd = Twist()

            cmd.linear.x = V
            cmd.angular.z = omega

            self.cmd_pub.publish(cmd)

        elif self.state == self.STATE_LINEAR:

            cmd = Twist()

            cmd.linear.x = self.cruise_speed
            cmd.angular.z = 0.0

            self.cmd_pub.publish(cmd)


class MultiCarNavigation:

    def __init__(self):

        rospy.init_node(
            "multi_car_navigation"
        )

        cars_str = rospy.get_param(
            "~cars"
        )

        target_xs_str = rospy.get_param(
            "~target_xs"
        )

        target_ys_str = rospy.get_param(
            "~target_ys"
        )

        desired_impact_time = rospy.get_param(
            "~desired_impact_time",
            14.0
        )

        self.cars = cars_str.split(",")

        target_xs = [
            float(x)
            for x in target_xs_str.split(",")
        ]

        target_ys = [
            float(y)
            for y in target_ys_str.split(",")
        ]

        self.controllers = []

        for i in range(len(self.cars)):

            self.controllers.append(

                CarController(
                    self.cars[i],
                    target_xs[i],
                    target_ys[i],
                    desired_impact_time
                )

            )

        rospy.loginfo(
            "Multi Car ITCG Initialized"
        )

    def run(self):

        rate = rospy.Rate(50)

        while not rospy.is_shutdown():

            all_ready = True
            all_reached = True

            for controller in self.controllers:

                controller.control_step()

                if not controller.pose_ready:
                    all_ready = False

                if not controller.reached:
                    all_reached = False

            if not all_ready:

                rospy.loginfo_throttle(
                    2.0,
                    "Waiting for mocap..."
                )

            if all_reached:

                rospy.loginfo(
                    "ALL VEHICLES REACHED TARGET"
                )

                rospy.signal_shutdown(
                    "mission complete"
                )

                break

            rate.sleep()


if __name__ == "__main__":

    try:

        node = MultiCarNavigation()

        node.run()

    except rospy.ROSInterruptException:
        pass