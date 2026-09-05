#!/usr/bin/env python3
# coding: utf-8

"""
Li et al. (CJA 2026) ITACG - latest constant-speed ROS hardware version.

Unified gate benchmark
----------------------
Designed task-start state:
    P0       = ( 2.0, 1.5 ) m
    theta0   = -90 deg

Gate:
    Pg       = (-1.5, 1.5 ) m
    theta_g  = 135 deg

Constant speed:
    V        = 0.23 m/s

Desired gate-crossing time:
    Td       = 20 s

Fixed offline cubic-Bezier design:
    R1 = 0.999848815 m
    R2 = 3.134360247 m

    P0 = ( 2.000000000,  1.500000000)
    P1 = ( 2.000000000,  0.500151185)
    P2 = ( 0.716327385, -0.716327385)
    P3 = (-1.500000000,  1.500000000)

Latest implementation choices
-----------------------------
1) Euclidean closest-point projection is used for reference association.
   This avoids the ill-conditioned x -> epsilon inversion near the
   vertical initial tangent.

2) The Li vector-field structure is retained:
       e_y = y_p - y
   and the practical hyperbolic-tangent vector field is used.

3) Meter-scale tracking gain:
       k1 = 1.0 1/m
   while
       k2 = 30, ka = 3.2, Ts = 8 s.

4) Hardware command channel:
       50 Hz
       |omega| <= 3 rad/s
       |dot(omega)| <= 2 rad/s^2

5) Optional additive command-channel disturbance:
       d_omega = +0.5 rad/s
       5 <= task_t <= 6 s

6) Strict task-start admissibility check.
   The 20-s task clock starts ONLY when the car crosses the designed
   start line and simultaneously satisfies:
       |start tangential position error| <= 0.02 m
       |start heading error|             <= 2 deg
       |start speed - 0.23|              <= 0.02 m/s
   Otherwise the trial is aborted and must be repeated.

Recommended physical placement
------------------------------
Before launching this node, place the car approximately at:
    x = 2.00 m
    y = 2.00 m
    yaw = -90 deg

Recommended practical tolerances for placement:
    x:   2.00 +/- 0.01 m
    y:   1.90 ~ 2.10 m
    yaw: -90 +/- 2 deg

The node accelerates during the run-in and automatically starts the
20-s Li task when crossing y=1.5 in the -y direction.

ROS defaults
------------
Pose:
    /vrpn_client_node/car1/pose
Odom:
    /car1/odom
Command:
    /car1/cmd_vel

Nominal:
    rosrun <your_package> li2026_itacg_constant_speed_ros_v2.py \
        _inject_disturbance:=false

Disturbance:
    rosrun <your_package> li2026_itacg_constant_speed_ros_v2.py \
        _inject_disturbance:=true
"""

import csv
import math
import os
from datetime import datetime

import rospy
import tf

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64


class LiITACGGuidance:
    def __init__(self):
        rospy.init_node(
            "li_2026_itacg_guidance_v2",
            anonymous=True,
        )

        # ============================================================
        # 1. ROS interfaces
        # ============================================================
        self.car_name = str(
            rospy.get_param(
                "~car_name",
                "car1",
            )
        )

        self.pose_topic = str(
            rospy.get_param(
                "~pose_topic",
                "/vrpn_client_node/{}/pose".format(
                    self.car_name
                ),
            )
        )

        self.odom_topic = str(
            rospy.get_param(
                "~odom_topic",
                "/{}/odom".format(
                    self.car_name
                ),
            )
        )

        self.cmd_topic = str(
            rospy.get_param(
                "~cmd_topic",
                "/{}/cmd_vel".format(
                    self.car_name
                ),
            )
        )

        self.telemetry_ns = str(
            rospy.get_param(
                "~telemetry_ns",
                "/li_itacg",
            )
        )

        # ============================================================
        # 2. Unified task
        # ============================================================
        self.start_x = float(
            rospy.get_param(
                "~start_x",
                2.0,
            )
        )
        self.start_y = float(
            rospy.get_param(
                "~start_y",
                1.5,
            )
        )
        self.start_yaw = math.radians(
            float(
                rospy.get_param(
                    "~start_yaw_deg",
                    -90.0,
                )
            )
        )

        self.target_x = float(
            rospy.get_param(
                "~target_x",
                -1.5,
            )
        )
        self.target_y = float(
            rospy.get_param(
                "~target_y",
                1.5,
            )
        )
        self.theta_g = math.radians(
            float(
                rospy.get_param(
                    "~theta_g_deg",
                    135.0,
                )
            )
        )

        self.desired_impact_time = float(
            rospy.get_param(
                "~T",
                20.0,
            )
        )

        self.constant_speed = float(
            rospy.get_param(
                "~v_const",
                0.23,
            )
        )

        # ============================================================
        # 3. Fixed offline Bezier design
        # ============================================================
        # These R1,R2 correspond to the unified task above.
        self.R1 = float(
            rospy.get_param(
                "~R1",
                0.999848815,
            )
        )
        self.R2 = float(
            rospy.get_param(
                "~R2",
                3.134360247,
            )
        )

        self.gamma1 = self.start_yaw
        self.gamma2 = self.normalize_angle(
            self.theta_g + math.pi
        )

        self.P0 = (
            self.start_x,
            self.start_y,
        )

        self.P1 = (
            self.P0[0]
            + self.R1
            * math.cos(self.gamma1),
            self.P0[1]
            + self.R1
            * math.sin(self.gamma1),
        )

        self.P3 = (
            self.target_x,
            self.target_y,
        )

        self.P2 = (
            self.P3[0]
            + self.R2
            * math.cos(self.gamma2),
            self.P3[1]
            + self.R2
            * math.sin(self.gamma2),
        )

        # ============================================================
        # 4. Li tracking-law parameters
        # ============================================================
        # Robot-scale value selected from the closest-point MATLAB sweep.
        self.k1 = float(
            rospy.get_param(
                "~k1",
                1.0,
            )
        )

        self.k2 = float(
            rospy.get_param(
                "~k2",
                30.0,
            )
        )

        self.ka = float(
            rospy.get_param(
                "~ka",
                3.2,
            )
        )

        self.Ts = float(
            rospy.get_param(
                "~Ts",
                8.0,
            )
        )

        # Practical hardware guard around t -> Ts^-.
        self.Ts_switch_margin = float(
            rospy.get_param(
                "~Ts_switch_margin",
                0.08,
            )
        )

        # ============================================================
        # 5. Physical command limits
        # ============================================================
        self.rate_hz = float(
            rospy.get_param(
                "~rate_hz",
                50.0,
            )
        )

        self.rate = rospy.Rate(
            self.rate_hz
        )

        self.max_angular = float(
            rospy.get_param(
                "~omega_max",
                3.0,
            )
        )

        self.max_yaw_accel = float(
            rospy.get_param(
                "~omega_slew",
                2.0,
            )
        )

        self.last_cmd_omega = 0.0

        # ============================================================
        # 6. Optional disturbance
        # ============================================================
        self.inject_disturbance = bool(
            rospy.get_param(
                "~inject_disturbance",
                True,
            )
        )

        self.dist_start_time = float(
            rospy.get_param(
                "~dist_start_time",
                5.0,
            )
        )

        self.dist_duration = float(
            rospy.get_param(
                "~dist_duration",
                1.0,
            )
        )

        self.dist_omega_val = float(
            rospy.get_param(
                "~dist_omega_val",
                0.5,
            )
        )

        # ============================================================
        # 7. Run-in and strict task-start acceptance
        # ============================================================
        self.runin_ramp_duration = float(
            rospy.get_param(
                "~runin_ramp_duration",
                0.50,
            )
        )

        self.runin_heading_gain = float(
            rospy.get_param(
                "~runin_heading_gain",
                1.2,
            )
        )

        # The node must start sufficiently before the start line.
        self.start_line_margin = float(
            rospy.get_param(
                "~start_line_margin",
                0.05,
            )
        )

        # Strict admissibility at the interpolated start-line crossing.
        self.start_tangent_tol = float(
            rospy.get_param(
                "~start_tangent_tol",
                0.02,
            )
        )

        self.start_yaw_tol = math.radians(
            float(
                rospy.get_param(
                    "~start_yaw_tol_deg",
                    2.0,
                )
            )
        )

        self.start_speed_tol = float(
            rospy.get_param(
                "~start_speed_tol",
                0.02,
            )
        )

        # ============================================================
        # 8. Runtime states
        # ============================================================
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.actual_speed = 0.0

        self.pose_ready = False
        self.odom_ready = False

        self.STATE_RUNIN = 1
        self.STATE_TRACK = 2
        self.STATE_TERMINAL = 3
        self.STATE_ABORT = 4

        self.state = self.STATE_RUNIN

        self.node_start_sec = None
        self.task_start_sec = None
        self.last_loop_sec = None

        # Start-crossing interpolation memory.
        self.prev_start_d = None
        self.prev_start_time_sec = None
        self.prev_start_x = None
        self.prev_start_y = None
        self.prev_start_yaw = None
        self.prev_start_speed = None

        # Gate-crossing interpolation memory.
        self.prev_gate_d = None
        self.prev_task_t = None
        self.prev_x = None
        self.prev_y = None
        self.prev_yaw = None
        self.prev_speed = None

        # Closest-point phase memory only for logging.
        self.last_epsilon = 0.0

        # ============================================================
        # 9. Precompute Bezier table for closest-point projection
        # ============================================================
        self.curve_N = int(
            rospy.get_param(
                "~curve_N",
                4001,
            )
        )

        self.eps_grid = []
        self.x_curve = []
        self.y_curve = []

        self.build_curve_lookup()

        # ============================================================
        # 10. CSV logger
        # ============================================================
        log_dir = os.path.expanduser("~")

        filename = datetime.now().strftime(
            "li2026_itacg_v2_%Y%m%d_%H%M%S.csv"
        )

        self.csv_path = os.path.join(
            log_dir,
            filename,
        )

        self.csv_file = open(
            self.csv_path,
            "w",
        )

        self.csv_writer = csv.writer(
            self.csv_file
        )

        self.csv_keys = [
            "node_time",
            "task_time",
            "state",
            "relative_dist",
            "epsilon_ref",
            "closest_dist",
            "x_ref",
            "y_ref",
            "theta_p",
            "theta_d",
            "ey",
            "theta_error",
            "a_p",
            "a_cmd",
            "omega_nominal",
            "dist_omega",
            "omega_requested",
            "omega_cmd",
            "speed_actual",
            "speed_desired",
            "yaw_actual",
            "pos_x",
            "pos_y",
            "gate_distance",
        ]

        angle_keys = {
            "theta_p",
            "theta_d",
            "theta_error",
            "yaw_actual",
        }

        headers = []

        for key in self.csv_keys:
            if key in angle_keys:
                headers.append(
                    key + "_deg"
                )
            else:
                headers.append(key)

        self.csv_writer.writerow(
            headers
        )

        rospy.on_shutdown(
            self.shutdown_hook
        )

        # ============================================================
        # 11. ROS interfaces and telemetry topics
        # ============================================================
        rospy.Subscriber(
            self.pose_topic,
            PoseStamped,
            self.pose_callback,
        )

        rospy.Subscriber(
            self.odom_topic,
            Odometry,
            self.odom_callback,
        )

        self.cmd_pub = rospy.Publisher(
            self.cmd_topic,
            Twist,
            queue_size=10,
        )

        topic_names = [
            "relative_dist",
            "epsilon_ref",
            "closest_dist",
            "x_ref",
            "y_ref",
            "theta_p",
            "theta_d",
            "ey",
            "theta_error",
            "a_p",
            "a_cmd",
            "omega_nominal",
            "dist_omega",
            "omega_requested",
            "omega_cmd",
            "speed_actual",
            "speed_desired",
            "yaw_actual",
            "gate_distance",
        ]

        self.pubs = {
            name: rospy.Publisher(
                self.telemetry_ns
                + "/"
                + name,
                Float64,
                queue_size=10,
            )
            for name in topic_names
        }

        rospy.loginfo(
            "Li ITACG v2 ready | "
            "closest-point | k1=%.3f 1/m | "
            "V=%.3f m/s | T=%.2f s | disturbance=%s",
            self.k1,
            self.constant_speed,
            self.desired_impact_time,
            str(self.inject_disturbance),
        )

        rospy.loginfo(
            "Strict start check: "
            "|e_tan|<=%.3f m, |e_yaw|<=%.2f deg, "
            "|e_v|<=%.3f m/s",
            self.start_tangent_tol,
            math.degrees(
                self.start_yaw_tol
            ),
            self.start_speed_tol,
        )

    # ================================================================
    # Utilities
    # ================================================================
    @staticmethod
    def sat(x, lo, hi):
        return max(
            lo,
            min(hi, x),
        )

    @staticmethod
    def normalize_angle(angle):
        return (
            angle + math.pi
        ) % (
            2.0 * math.pi
        ) - math.pi

    @staticmethod
    def lerp_angle(
        a0,
        a1,
        r,
    ):
        return LiITACGGuidance.normalize_angle(
            a0
            + r
            * LiITACGGuidance.normalize_angle(
                a1 - a0
            )
        )

    @staticmethod
    def sech_sq(x):
        if abs(x) > 40.0:
            return 0.0

        c = math.cosh(x)

        return 1.0 / (
            c * c
        )

    def slew_limit(
        self,
        x,
        x_last,
        rate,
        dt,
    ):
        return (
            x_last
            + self.sat(
                x - x_last,
                -rate * dt,
                rate * dt,
            )
        )

    # ================================================================
    # ROS callbacks
    # ================================================================
    def pose_callback(
        self,
        msg,
    ):
        self.current_x = (
            msg.pose.position.x
        )
        self.current_y = (
            msg.pose.position.y
        )

        q = msg.pose.orientation

        _, _, self.current_yaw = (
            tf.transformations.euler_from_quaternion(
                (
                    q.x,
                    q.y,
                    q.z,
                    q.w,
                )
            )
        )

        self.pose_ready = True

    def odom_callback(
        self,
        msg,
    ):
        self.actual_speed = (
            math.hypot(
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
            )
        )

        self.odom_ready = True

    # ================================================================
    # Bezier geometry
    # ================================================================
    def bezier_state(
        self,
        eps,
    ):
        eps = self.sat(
            eps,
            0.0,
            1.0,
        )

        q = 1.0 - eps

        P0x, P0y = self.P0
        P1x, P1y = self.P1
        P2x, P2y = self.P2
        P3x, P3y = self.P3

        x = (
            q ** 3 * P0x
            + 3.0
            * eps
            * q ** 2
            * P1x
            + 3.0
            * eps ** 2
            * q
            * P2x
            + eps ** 3
            * P3x
        )

        y = (
            q ** 3 * P0y
            + 3.0
            * eps
            * q ** 2
            * P1y
            + 3.0
            * eps ** 2
            * q
            * P2y
            + eps ** 3
            * P3y
        )

        dx = (
            3.0
            * q ** 2
            * (P1x - P0x)
            + 6.0
            * q
            * eps
            * (P2x - P1x)
            + 3.0
            * eps ** 2
            * (P3x - P2x)
        )

        dy = (
            3.0
            * q ** 2
            * (P1y - P0y)
            + 6.0
            * q
            * eps
            * (P2y - P1y)
            + 3.0
            * eps ** 2
            * (P3y - P2y)
        )

        ddx = (
            6.0
            * q
            * (
                P2x
                - 2.0 * P1x
                + P0x
            )
            + 6.0
            * eps
            * (
                P3x
                - 2.0 * P2x
                + P1x
            )
        )

        ddy = (
            6.0
            * q
            * (
                P2y
                - 2.0 * P1y
                + P0y
            )
            + 6.0
            * eps
            * (
                P3y
                - 2.0 * P2y
                + P1y
            )
        )

        theta_p = math.atan2(
            dy,
            dx,
        )

        den = (
            dx * dx
            + dy * dy
        ) ** 1.5

        if den < 1.0e-14:
            kappa = 0.0
        else:
            kappa = (
                dx * ddy
                - dy * ddx
            ) / den

        a_p = (
            self.constant_speed
            * self.constant_speed
            * kappa
        )

        return {
            "x": x,
            "y": y,
            "theta_p": theta_p,
            "kappa": kappa,
            "a_p": a_p,
        }

    def build_curve_lookup(
        self,
    ):
        self.eps_grid = []
        self.x_curve = []
        self.y_curve = []

        for i in range(
            self.curve_N
        ):
            eps = float(i) / float(
                self.curve_N - 1
            )

            ref = self.bezier_state(
                eps
            )

            self.eps_grid.append(
                eps
            )
            self.x_curve.append(
                ref["x"]
            )
            self.y_curve.append(
                ref["y"]
            )

    def closest_point_parameter(
        self,
        x,
        y,
    ):
        best_i = 0
        best_d2 = float("inf")

        for i in range(
            self.curve_N
        ):
            dx = (
                self.x_curve[i] - x
            )
            dy = (
                self.y_curve[i] - y
            )

            d2 = (
                dx * dx
                + dy * dy
            )

            if d2 < best_d2:
                best_d2 = d2
                best_i = i

        eps = self.eps_grid[
            best_i
        ]

        closest_dist = math.sqrt(
            best_d2
        )

        return (
            eps,
            closest_dist,
        )

    # ================================================================
    # Start / gate line geometry
    # ================================================================
    def start_line_distance(
        self,
        x,
        y,
    ):
        return (
            math.cos(
                self.start_yaw
            )
            * (
                x - self.start_x
            )
            + math.sin(
                self.start_yaw
            )
            * (
                y - self.start_y
            )
        )

    def start_tangent_error(
        self,
        x,
        y,
    ):
        tx = -math.sin(
            self.start_yaw
        )
        ty = math.cos(
            self.start_yaw
        )

        return (
            tx
            * (
                x - self.start_x
            )
            + ty
            * (
                y - self.start_y
            )
        )

    def gate_signed_distance(
        self,
        x,
        y,
    ):
        return (
            math.cos(
                self.theta_g
            )
            * (
                x - self.target_x
            )
            + math.sin(
                self.theta_g
            )
            * (
                y - self.target_y
            )
        )

    # ================================================================
    # Li tracking law
    # ================================================================
    def calc_li_command(
        self,
        task_t,
    ):
        eps, closest_dist = (
            self.closest_point_parameter(
                self.current_x,
                self.current_y,
            )
        )

        self.last_epsilon = eps

        ref = self.bezier_state(
            eps
        )

        theta_p = ref[
            "theta_p"
        ]

        a_p = ref[
            "a_p"
        ]

        # Paper Eq. (26), retained exactly in structure.
        ey = (
            ref["y"]
            - self.current_y
        )

        theta_p_dot = (
            a_p
            / self.constant_speed
        )

        # Paper Eq. (27)
        ey_dot = (
            self.constant_speed
            * math.sin(
                theta_p
            )
            - self.constant_speed
            * math.sin(
                self.current_yaw
            )
        )

        # Practical smooth Eq. (28).
        tanh_1 = math.tanh(
            self.k1 * ey
        )

        tanh_2 = math.tanh(
            self.k2
            * math.cos(
                theta_p
            )
        )

        theta_d = (
            self.normalize_angle(
                theta_p
                + 0.5
                * math.pi
                * tanh_1
                * tanh_2
            )
        )

        # Paper Eq. (29).
        theta_d_dot = (
            theta_p_dot
            + self.k1
            * 0.5
            * math.pi
            * self.sech_sq(
                self.k1 * ey
            )
            * ey_dot
            * tanh_2
            - self.k2
            * 0.5
            * math.pi
            * tanh_1
            * self.sech_sq(
                self.k2
                * math.cos(
                    theta_p
                )
            )
            * math.sin(
                theta_p
            )
            * theta_p_dot
        )

        # Paper Eq. (36).
        theta_error = (
            self.normalize_angle(
                self.current_yaw
                - theta_d
            )
        )

        # Paper Eq. (37) with practical switch guard.
        if task_t < (
            self.Ts
            - self.Ts_switch_margin
        ):
            den = max(
                self.Ts - task_t,
                self.Ts_switch_margin,
            )

            a_cmd = (
                self.constant_speed
                * (
                    -self.ka
                    / den
                    * theta_error
                    + theta_d_dot
                )
            )

        else:
            a_cmd = (
                self.constant_speed
                * (
                    -self.ka
                    * theta_error
                    + theta_d_dot
                )
            )

        omega_nominal = (
            a_cmd
            / self.constant_speed
        )

        R = math.hypot(
            self.target_x
            - self.current_x,
            self.target_y
            - self.current_y,
        )

        data = {
            "R": R,
            "epsilon_ref": eps,
            "closest_dist":
                closest_dist,
            "x_ref": ref["x"],
            "y_ref": ref["y"],
            "theta_p": theta_p,
            "theta_d": theta_d,
            "ey": ey,
            "theta_error":
                theta_error,
            "a_p": a_p,
            "a_cmd": a_cmd,
            "omega_nominal":
                omega_nominal,
        }

        return (
            omega_nominal,
            data,
        )

    # ================================================================
    # Command and logging
    # ================================================================
    def publish_cmd(
        self,
        v,
        omega_requested,
        dt,
    ):
        omega_limited = (
            self.slew_limit(
                omega_requested,
                self.last_cmd_omega,
                self.max_yaw_accel,
                dt,
            )
        )

        omega_cmd = self.sat(
            omega_limited,
            -self.max_angular,
            self.max_angular,
        )

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = (
            omega_cmd
        )

        self.last_cmd_omega = (
            omega_cmd
        )

        self.cmd_pub.publish(
            cmd
        )

        return omega_cmd

    def publish_stop(
        self,
    ):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0

        self.last_cmd_omega = 0.0

        self.cmd_pub.publish(
            cmd
        )

    def publish_telemetry(
        self,
        data,
    ):
        angle_keys = {
            "theta_p",
            "theta_d",
            "theta_error",
            "yaw_actual",
        }

        for key, val in (
            data.items()
        ):
            if key not in self.pubs:
                continue

            if key in angle_keys:
                self.pubs[
                    key
                ].publish(
                    math.degrees(
                        val
                    )
                )
            else:
                self.pubs[
                    key
                ].publish(
                    val
                )

    def record_csv(
        self,
        data,
    ):
        angle_keys = {
            "theta_p",
            "theta_d",
            "theta_error",
            "yaw_actual",
        }

        row = []

        for key in self.csv_keys:
            val = data.get(
                key,
                0.0,
            )

            if key in angle_keys:
                val = math.degrees(
                    val
                )

            row.append(
                val
            )

        self.csv_writer.writerow(
            row
        )

    def shutdown_hook(
        self,
    ):
        try:
            self.publish_stop()
        except Exception:
            pass

        if (
            hasattr(
                self,
                "csv_file",
            )
            and not self.csv_file.closed
        ):
            self.csv_file.close()

            rospy.loginfo(
                "CSV saved to: %s",
                self.csv_path,
            )

    # ================================================================
    # Strict start-line trigger
    # ================================================================
    def update_start_memory(
        self,
        d,
        now_sec,
    ):
        self.prev_start_d = d
        self.prev_start_time_sec = (
            now_sec
        )
        self.prev_start_x = (
            self.current_x
        )
        self.prev_start_y = (
            self.current_y
        )
        self.prev_start_yaw = (
            self.current_yaw
        )
        self.prev_start_speed = (
            self.actual_speed
        )

    def check_task_start(
        self,
        now_sec,
    ):
        d = self.start_line_distance(
            self.current_x,
            self.current_y,
        )

        if self.prev_start_d is None:
            self.update_start_memory(
                d,
                now_sec,
            )
            return False

        crossed = (
            self.prev_start_d < 0.0
            and d >= 0.0
        )

        if not crossed:
            self.update_start_memory(
                d,
                now_sec,
            )
            return False

        den = (
            d - self.prev_start_d
        )

        if abs(den) < 1.0e-12:
            alpha = 0.0
        else:
            alpha = (
                -self.prev_start_d
                / den
            )

        alpha = self.sat(
            alpha,
            0.0,
            1.0,
        )

        t_cross = (
            self.prev_start_time_sec
            + alpha
            * (
                now_sec
                - self.prev_start_time_sec
            )
        )

        x_cross = (
            self.prev_start_x
            + alpha
            * (
                self.current_x
                - self.prev_start_x
            )
        )

        y_cross = (
            self.prev_start_y
            + alpha
            * (
                self.current_y
                - self.prev_start_y
            )
        )

        yaw_cross = self.lerp_angle(
            self.prev_start_yaw,
            self.current_yaw,
            alpha,
        )

        speed_cross = (
            self.prev_start_speed
            + alpha
            * (
                self.actual_speed
                - self.prev_start_speed
            )
        )

        e_tan = self.start_tangent_error(
            x_cross,
            y_cross,
        )

        e_yaw = self.normalize_angle(
            yaw_cross
            - self.start_yaw
        )

        e_speed = (
            speed_cross
            - self.constant_speed
        )

        rospy.loginfo(
            "========== START-LINE CROSSING =========="
        )
        rospy.loginfo(
            "pose: x=%.6f, y=%.6f, yaw=%.4f deg",
            x_cross,
            y_cross,
            math.degrees(
                yaw_cross
            ),
        )
        rospy.loginfo(
            "speed=%.5f m/s",
            speed_cross,
        )
        rospy.loginfo(
            "e_tan=%+.5f m | e_yaw=%+.4f deg | e_v=%+.5f m/s",
            e_tan,
            math.degrees(
                e_yaw
            ),
            e_speed,
        )

        valid = (
            abs(e_tan)
            <= self.start_tangent_tol
            and abs(e_yaw)
            <= self.start_yaw_tol
            and abs(e_speed)
            <= self.start_speed_tol
        )

        if not valid:
            rospy.logerr(
                "START REJECTED. "
                "Required: |e_tan|<=%.3f m, "
                "|e_yaw|<=%.2f deg, "
                "|e_v|<=%.3f m/s",
                self.start_tangent_tol,
                math.degrees(
                    self.start_yaw_tol
                ),
                self.start_speed_tol,
            )

            self.state = (
                self.STATE_ABORT
            )

            self.publish_stop()

            return False

        # Valid start: define the exact interpolated task-clock zero.
        self.task_start_sec = (
            t_cross
        )
        self.state = (
            self.STATE_TRACK
        )

        # Gate-crossing memory starts from current measured state.
        current_task_t = (
            now_sec
            - self.task_start_sec
        )

        self.prev_gate_d = (
            self.gate_signed_distance(
                self.current_x,
                self.current_y,
            )
        )

        self.prev_task_t = (
            current_task_t
        )
        self.prev_x = (
            self.current_x
        )
        self.prev_y = (
            self.current_y
        )
        self.prev_yaw = (
            self.current_yaw
        )
        self.prev_speed = (
            self.actual_speed
        )

        # Do not carry a run-in yaw command transient into the Li task.
        self.last_cmd_omega = 0.0

        rospy.loginfo(
            "START ACCEPTED. "
            "Li 20-s task clock begins at interpolated crossing."
        )
        rospy.loginfo(
            "========================================="
        )

        return True

    # ================================================================
    # First gate-line crossing
    # ================================================================
    def update_gate_memory(
        self,
        d,
        task_t,
    ):
        self.prev_gate_d = d
        self.prev_task_t = task_t
        self.prev_x = self.current_x
        self.prev_y = self.current_y
        self.prev_yaw = (
            self.current_yaw
        )
        self.prev_speed = (
            self.actual_speed
        )

    def check_first_gate_crossing(
        self,
        task_t,
    ):
        d = self.gate_signed_distance(
            self.current_x,
            self.current_y,
        )

        if self.prev_gate_d is None:
            self.update_gate_memory(
                d,
                task_t,
            )
            return False

        crossed = (
            self.prev_gate_d < 0.0
            and d >= 0.0
        )

        if not crossed:
            self.update_gate_memory(
                d,
                task_t,
            )
            return False

        den = (
            d - self.prev_gate_d
        )

        if abs(den) < 1.0e-12:
            alpha = 0.0
        else:
            alpha = (
                -self.prev_gate_d
                / den
            )

        alpha = self.sat(
            alpha,
            0.0,
            1.0,
        )

        t_cross = (
            self.prev_task_t
            + alpha
            * (
                task_t
                - self.prev_task_t
            )
        )

        x_cross = (
            self.prev_x
            + alpha
            * (
                self.current_x
                - self.prev_x
            )
        )

        y_cross = (
            self.prev_y
            + alpha
            * (
                self.current_y
                - self.prev_y
            )
        )

        yaw_cross = self.lerp_angle(
            self.prev_yaw,
            self.current_yaw,
            alpha,
        )

        speed_cross = (
            self.prev_speed
            + alpha
            * (
                self.actual_speed
                - self.prev_speed
            )
        )

        gate_tx = -math.sin(
            self.theta_g
        )
        gate_ty = math.cos(
            self.theta_g
        )

        e_p = (
            gate_tx
            * (
                x_cross
                - self.target_x
            )
            + gate_ty
            * (
                y_cross
                - self.target_y
            )
        )

        e_theta = (
            self.normalize_angle(
                yaw_cross
                - self.theta_g
            )
        )

        e_T = (
            t_cross
            - self.desired_impact_time
        )

        e_v = (
            speed_cross
            - self.constant_speed
        )

        rospy.loginfo(
            "========== FIRST GATE CROSSING =========="
        )
        rospy.loginfo(
            "t_cross=%.6f s | e_T=%+.6f s",
            t_cross,
            e_T,
        )
        rospy.loginfo(
            "e_p=%+.6f m | e_theta=%+.6f deg",
            e_p,
            math.degrees(
                e_theta
            ),
        )
        rospy.loginfo(
            "v_cross=%.6f m/s | e_v=%+.6f m/s",
            speed_cross,
            e_v,
        )
        rospy.loginfo(
            "disturbance enabled=%s",
            str(
                self.inject_disturbance
            ),
        )
        rospy.loginfo(
            "========================================="
        )

        return True

    # ================================================================
    # Main
    # ================================================================
    def run(
        self,
    ):
        rospy.loginfo(
            "Waiting for motion-capture pose..."
        )

        try:
            pose_msg = (
                rospy.wait_for_message(
                    self.pose_topic,
                    PoseStamped,
                    timeout=5.0,
                )
            )

            self.pose_callback(
                pose_msg
            )

        except rospy.ROSException:
            rospy.logerr(
                "No pose data. Node exits."
            )
            return

        rospy.loginfo(
            "Waiting for odometry..."
        )

        try:
            odom_msg = (
                rospy.wait_for_message(
                    self.odom_topic,
                    Odometry,
                    timeout=5.0,
                )
            )

            self.odom_callback(
                odom_msg
            )

        except rospy.ROSException:
            rospy.logerr(
                "No odometry data. Node exits."
            )
            return

        now = rospy.Time.now()

        self.node_start_sec = (
            now.to_sec()
        )
        self.last_loop_sec = (
            self.node_start_sec
        )

        initial_start_d = (
            self.start_line_distance(
                self.current_x,
                self.current_y,
            )
        )

        initial_e_tan = (
            self.start_tangent_error(
                self.current_x,
                self.current_y,
            )
        )

        rospy.loginfo(
            "Initial pose: x=%.4f, y=%.4f, yaw=%.2f deg",
            self.current_x,
            self.current_y,
            math.degrees(
                self.current_yaw
            ),
        )

        if initial_start_d > (
            -self.start_line_margin
        ):
            rospy.logerr(
                "Car is not sufficiently before the start line. "
                "signed distance=%.3f m. "
                "Place near (2.0,2.0), yaw=-90 deg.",
                initial_start_d,
            )
            self.publish_stop()
            return

        if abs(initial_e_tan) > 0.05:
            rospy.logwarn(
                "Initial x/tangential offset is %.3f m. "
                "For a clean nominal trial, reposition closer to x=2.0.",
                initial_e_tan,
            )

        rospy.loginfo(
            "Run-in begins. "
            "The 20-s task clock has NOT started yet."
        )

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            now_sec = now.to_sec()

            dt_real = (
                now_sec
                - self.last_loop_sec
            )

            if (
                dt_real <= 0.001
                or dt_real > 0.2
            ):
                dt_real = (
                    1.0
                    / self.rate_hz
                )

            self.last_loop_sec = (
                now_sec
            )

            node_t = (
                now_sec
                - self.node_start_sec
            )

            # --------------------------------------------------------
            # RUN-IN
            # --------------------------------------------------------
            if self.state == self.STATE_RUNIN:
                ramp_ratio = self.sat(
                    node_t
                    / max(
                        self.runin_ramp_duration,
                        1.0e-6,
                    ),
                    0.0,
                    1.0,
                )

                V_cmd = (
                    self.constant_speed
                    * ramp_ratio
                )

                yaw_error = (
                    self.normalize_angle(
                        self.start_yaw
                        - self.current_yaw
                    )
                )

                omega_requested = (
                    self.runin_heading_gain
                    * yaw_error
                )

                omega_cmd = (
                    self.publish_cmd(
                        V_cmd,
                        omega_requested,
                        dt_real,
                    )
                )

                telemetry = {
                    "node_time":
                        node_t,
                    "task_time":
                        0.0,
                    "state":
                        self.STATE_RUNIN,
                    "relative_dist":
                        math.hypot(
                            self.target_x
                            - self.current_x,
                            self.target_y
                            - self.current_y,
                        ),
                    "omega_nominal":
                        omega_requested,
                    "dist_omega":
                        0.0,
                    "omega_requested":
                        omega_requested,
                    "omega_cmd":
                        omega_cmd,
                    "speed_actual":
                        self.actual_speed,
                    "speed_desired":
                        V_cmd,
                    "yaw_actual":
                        self.current_yaw,
                    "pos_x":
                        self.current_x,
                    "pos_y":
                        self.current_y,
                    "gate_distance":
                        self.gate_signed_distance(
                            self.current_x,
                            self.current_y,
                        ),
                }

                self.record_csv(
                    telemetry
                )

                started = (
                    self.check_task_start(
                        now_sec
                    )
                )

                if self.state == self.STATE_ABORT:
                    break

                if started:
                    self.rate.sleep()
                    continue

            # --------------------------------------------------------
            # LI TRACKING
            # --------------------------------------------------------
            elif self.state == self.STATE_TRACK:
                task_t = (
                    now_sec
                    - self.task_start_sec
                )

                (
                    omega_nominal,
                    data,
                ) = self.calc_li_command(
                    task_t
                )

                dist_omega = 0.0

                if (
                    self.inject_disturbance
                    and self.dist_start_time
                    <= task_t
                    <= (
                        self.dist_start_time
                        + self.dist_duration
                    )
                ):
                    dist_omega = (
                        self.dist_omega_val
                    )

                    rospy.logwarn_throttle(
                        0.1,
                        "DISTURBANCE: t=%.2f s, d_omega=%+.2f rad/s",
                        task_t,
                        dist_omega,
                    )

                omega_requested = (
                    omega_nominal
                    + dist_omega
                )

                omega_cmd = (
                    self.publish_cmd(
                        self.constant_speed,
                        omega_requested,
                        dt_real,
                    )
                )

                gate_d = (
                    self.gate_signed_distance(
                        self.current_x,
                        self.current_y,
                    )
                )

                telemetry = {
                    "node_time":
                        node_t,
                    "task_time":
                        task_t,
                    "state":
                        self.STATE_TRACK,
                    "relative_dist":
                        data["R"],
                    "epsilon_ref":
                        data[
                            "epsilon_ref"
                        ],
                    "closest_dist":
                        data[
                            "closest_dist"
                        ],
                    "x_ref":
                        data["x_ref"],
                    "y_ref":
                        data["y_ref"],
                    "theta_p":
                        data["theta_p"],
                    "theta_d":
                        data["theta_d"],
                    "ey":
                        data["ey"],
                    "theta_error":
                        data[
                            "theta_error"
                        ],
                    "a_p":
                        data["a_p"],
                    "a_cmd":
                        data["a_cmd"],
                    "omega_nominal":
                        omega_nominal,
                    "dist_omega":
                        dist_omega,
                    "omega_requested":
                        omega_requested,
                    "omega_cmd":
                        omega_cmd,
                    "speed_actual":
                        self.actual_speed,
                    "speed_desired":
                        self.constant_speed,
                    "yaw_actual":
                        self.current_yaw,
                    "pos_x":
                        self.current_x,
                    "pos_y":
                        self.current_y,
                    "gate_distance":
                        gate_d,
                }

                self.publish_telemetry(
                    telemetry
                )

                self.record_csv(
                    telemetry
                )

                if (
                    self.check_first_gate_crossing(
                        task_t
                    )
                ):
                    self.state = (
                        self.STATE_TERMINAL
                    )
                    self.publish_stop()
                    break

                if task_t > (
                    self.desired_impact_time
                    + 5.0
                ):
                    rospy.logerr(
                        "Timeout: no gate crossing by T+5 s."
                    )
                    self.state = (
                        self.STATE_ABORT
                    )
                    self.publish_stop()
                    break

            # --------------------------------------------------------
            # TERMINAL / ABORT
            # --------------------------------------------------------
            elif (
                self.state
                == self.STATE_TERMINAL
            ):
                self.publish_stop()
                break

            elif (
                self.state
                == self.STATE_ABORT
            ):
                self.publish_stop()
                break

            self.rate.sleep()


if __name__ == "__main__":
    try:
        LiITACGGuidance().run()

    except rospy.ROSInterruptException:
        pass
