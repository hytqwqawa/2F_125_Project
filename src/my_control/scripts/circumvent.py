#!/usr/bin/env python3
# coding: utf-8

import math
import rospy
import tf
from geometry_msgs.msg import PoseStamped, Twist


class AckermannCarController:
    """One Ackermann car controller using the same control law/topics as navigation_to_exit.py."""

    def __init__(self, car_name, target_x, target_y, cruise_speed, common):
        self.car_name = car_name
        self.target_x = target_x
        self.target_y = target_y
        self.cruise_speed = cruise_speed

        # Keep exactly the same topic convention as the original single-car node.
        self.pose_topic = "/vrpn_client_node/{}/pose".format(car_name)
        self.cmd_topic = "/{}/cmd_vel".format(car_name)

        # Common safety/control parameters, inherited from the original program.
        self.bound_x_min = common["bound_x_min"]
        self.bound_x_max = common["bound_x_max"]
        self.bound_y_min = common["bound_y_min"]
        self.bound_y_max = common["bound_y_max"]
        self.safety_margin = common["safety_margin"]
        self.boundary_predict_time = common["boundary_predict_time"]

        self.min_speed = common["min_speed"]
        self.slow_radius = common["slow_radius"]
        self.k_yaw = common["k_yaw"]
        self.max_angular = common["max_angular"]
        self.distance_threshold = common["distance_threshold"]
        self.allow_reverse = common["allow_reverse"]
        self.angular_cmd_is_steering = common["angular_cmd_is_steering"]

        self.switch_to_reverse_angle = math.radians(110.0)
        self.switch_to_forward_angle = math.radians(70.0)

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.pose_ready = False
        self.direction = 1
        self.finished = False
        self.failed = False

        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=10)
        self.pose_sub = rospy.Subscriber(
            self.pose_topic,
            PoseStamped,
            self.pose_callback,
            queue_size=1,
        )

    def pose_callback(self, msg):
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y

        quat = (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        )
        _, _, self.current_yaw = tf.transformations.euler_from_quaternion(quat)
        self.pose_ready = True

    @staticmethod
    def normalize_angle(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def sat(value, lower, upper):
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
            self.bound_x_min + self.safety_margin
            <= x
            <= self.bound_x_max - self.safety_margin
            and self.bound_y_min + self.safety_margin
            <= y
            <= self.bound_y_max - self.safety_margin
        )

    def target_is_valid(self):
        return self.inside_safe_area(self.target_x, self.target_y)

    def current_touch_boundary(self):
        return not self.inside_safe_area(self.current_x, self.current_y)

    def predict_touch_boundary(self, v):
        x_next = (
            self.current_x
            + v * math.cos(self.current_yaw) * self.boundary_predict_time
        )
        y_next = (
            self.current_y
            + v * math.sin(self.current_yaw) * self.boundary_predict_time
        )
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

        # Same turning slowdown used in the original code.
        turn_slow = max(
            0.25,
            math.cos(min(abs(heading_error), math.pi / 2.0)),
        )
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

    def publish_stop(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

    def update_and_publish(self):
        """Run one control iteration for this car."""
        if self.finished or self.failed:
            self.publish_stop()
            return

        dist = self.distance_to_target()

        if dist <= self.distance_threshold:
            self.finished = True
            self.publish_stop()
            rospy.loginfo(
                "[%s] Target reached. pos=(%.3f, %.3f), target=(%.3f, %.3f), dist=%.3f",
                self.car_name,
                self.current_x,
                self.current_y,
                self.target_x,
                self.target_y,
                dist,
            )
            return

        if self.current_touch_boundary():
            self.failed = True
            self.publish_stop()
            rospy.logwarn(
                "[%s] Safety boundary reached. This car is stopped.",
                self.car_name,
            )
            return

        cmd, dist, heading_error = self.compute_cmd()

        if self.predict_touch_boundary(cmd.linear.x):
            self.failed = True
            self.publish_stop()
            rospy.logwarn(
                "[%s] Predicted safety-boundary violation. This car is stopped.",
                self.car_name,
            )
            return

        self.cmd_pub.publish(cmd)
        rospy.loginfo_throttle(
            0.5,
            "[%s] dist=%.3f, v=%.3f, omega=%.3f",
            self.car_name,
            dist,
            cmd.linear.x,
            cmd.angular.z,
        )


class ThreeCarNavigation:
    def __init__(self):
        rospy.init_node("navigation_three_cars", anonymous=False)

        common = {
            "bound_x_min": rospy.get_param("~bound_x_min", -2.3),
            "bound_x_max": rospy.get_param("~bound_x_max", 3.6),
            "bound_y_min": rospy.get_param("~bound_y_min", -1.9),
            "bound_y_max": rospy.get_param("~bound_y_max", 1.7),
            "safety_margin": rospy.get_param("~safety_margin", 0.08),
            "boundary_predict_time": rospy.get_param("~boundary_predict_time", 0.25),
            "min_speed": rospy.get_param("~min_speed", 0.15),
            "slow_radius": rospy.get_param("~slow_radius", 0.6),
            "k_yaw": rospy.get_param("~k_yaw", 2.2),
            "max_angular": rospy.get_param("~max_angular", 1.6),
            "distance_threshold": rospy.get_param("~distance_threshold", 0.03),
            "allow_reverse": rospy.get_param("~allow_reverse", True),
            "angular_cmd_is_steering": rospy.get_param(
                "~angular_cmd_is_steering", False
            ),
        }

        self.cars = [
            AckermannCarController(
                "car2",
                rospy.get_param("~car2_target_x", 3.0),
                rospy.get_param("~car2_target_y", -1.0),
                rospy.get_param("~car2_speed", 0.8),
                common,
            ),
            AckermannCarController(
                "car3",
                rospy.get_param("~car3_target_x", 2.5),
                rospy.get_param("~car3_target_y", 0.0),
                rospy.get_param("~car3_speed", 0.8),
                common,
            ),
            AckermannCarController(
                "car4",
                rospy.get_param("~car4_target_x", 3.0),
                rospy.get_param("~car4_target_y", 1.0),
                rospy.get_param("~car4_speed", 0.8),
                common,
            ),
        ]

        self.rate = rospy.Rate(50)
        rospy.on_shutdown(self.stop_all)

        rospy.loginfo("Three-car Ackermann navigation initialized.")
        for car in self.cars:
            rospy.loginfo(
                "[%s] pose=%s, cmd=%s, target=(%.3f, %.3f), cruise_speed=%.3f",
                car.car_name,
                car.pose_topic,
                car.cmd_topic,
                car.target_x,
                car.target_y,
                car.cruise_speed,
            )

    def stop_all(self):
        for _ in range(3):
            for car in self.cars:
                car.publish_stop()
            rospy.sleep(0.02)

    def all_pose_ready(self):
        return all(car.pose_ready for car in self.cars)

    def validate_targets(self):
        ok = True
        for car in self.cars:
            if not car.target_is_valid():
                rospy.logerr(
                    "[%s] Target (%.3f, %.3f) is outside the configured safe area.",
                    car.car_name,
                    car.target_x,
                    car.target_y,
                )
                ok = False
        return ok

    def wait_for_all_poses(self, timeout=5.0):
        rospy.loginfo("Waiting for car2/car3/car4 VRPN poses before simultaneous start...")
        start_time = rospy.Time.now()

        while not rospy.is_shutdown():
            if self.all_pose_ready():
                return True

            if (rospy.Time.now() - start_time).to_sec() > timeout:
                missing = [car.car_name for car in self.cars if not car.pose_ready]
                rospy.logerr("Pose timeout. Missing VRPN pose: %s", ", ".join(missing))
                return False

            # Ensure nobody moves before all three pose streams are ready.
            for car in self.cars:
                car.publish_stop()
            self.rate.sleep()

        return False

    def run(self):
        if not self.validate_targets():
            self.stop_all()
            return

        if not self.wait_for_all_poses(timeout=5.0):
            self.stop_all()
            return

        rospy.loginfo("All three poses are ready. car2/car3/car4 START NOW.")

        while not rospy.is_shutdown():
            for car in self.cars:
                car.update_and_publish()

            # Each car can arrive at a different time. Exit only when all are done/stopped.
            if all(car.finished or car.failed for car in self.cars):
                rospy.loginfo("All three cars have finished or been safety-stopped.")
                self.stop_all()
                break

            self.rate.sleep()


if __name__ == "__main__":
    try:
        node = ThreeCarNavigation()
        node.run()
    except rospy.ROSInterruptException:
        pass
