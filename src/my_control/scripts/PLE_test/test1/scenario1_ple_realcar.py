#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scenario 1: real-car execution of the accepted actuator-aware PLE reference.

The reference and actuator-profile CSV files are exported from the accepted
MATLAB online-v2 result. This node executes the selected candidate on car4.
It does not solve the bridge online; it validates the real-car execution layer.
It validates:

1. tracking of the complete accepted composite reference;
2. continuous local Frenet projection across both interfaces;
3. the scheduled gamma ramp;
4. bridge contraction and terminal insertion;
5. real vehicle curvature demand and certificate behavior.

Default topics are configured for car4.
"""

import csv
import math
import os
import threading
import time
from datetime import datetime

import numpy as np
import rospy
import tf
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def normalize_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def smoothstep01(value):
    value = clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


class ReferenceTable:
    REQUIRED = (
        "s_ref", "x_ref", "y_ref", "theta_ref",
        "kappa_ref", "kappa_prime_ref", "gamma", "phase",
        "s_disc", "s_r", "s_splice", "s_in",
    )

    def __init__(self, filename):
        filename = os.path.expanduser(filename)
        if not os.path.isfile(filename):
            raise IOError("Reference CSV does not exist: {}".format(filename))

        columns = {name: [] for name in self.REQUIRED}
        with open(filename, "r") as file_object:
            reader = csv.DictReader(file_object)
            if reader.fieldnames is None:
                raise ValueError("Reference CSV has no header.")
            missing = [
                name for name in self.REQUIRED
                if name not in reader.fieldnames
            ]
            if missing:
                raise ValueError(
                    "Reference CSV is missing columns: {}".format(missing)
                )
            if "beta_nominal" in reader.fieldnames:
                beta_name = "beta_nominal"
            elif "beta" in reader.fieldnames:
                beta_name = "beta"
            else:
                raise ValueError(
                    "Reference CSV must contain beta_nominal or beta."
                )

            columns["beta"] = []
            for row in reader:
                for name in self.REQUIRED:
                    if name == "phase":
                        columns[name].append(row[name])
                    else:
                        columns[name].append(float(row[name]))
                columns["beta"].append(float(row[beta_name]))

        if len(columns["s_ref"]) < 20:
            raise ValueError("Reference CSV contains too few samples.")

        self.filename = filename
        self.s = np.asarray(columns["s_ref"], dtype=float)
        self.x = np.asarray(columns["x_ref"], dtype=float)
        self.y = np.asarray(columns["y_ref"], dtype=float)
        self.theta = np.asarray(columns["theta_ref"], dtype=float)
        self.kappa = np.asarray(columns["kappa_ref"], dtype=float)
        self.kappa_prime = np.asarray(
            columns["kappa_prime_ref"], dtype=float
        )
        self.gamma = np.asarray(columns["gamma"], dtype=float)
        self.beta = np.asarray(columns["beta"], dtype=float)
        self.phase = list(columns["phase"])

        if not np.all(np.diff(self.s) > 0.0):
            raise ValueError("Reference coordinate s_ref must be increasing.")
        for array_name in (
            "x", "y", "theta", "kappa", "kappa_prime", "gamma", "beta"
        ):
            if not np.all(np.isfinite(getattr(self, array_name))):
                raise ValueError(
                    "Reference column {} contains nonfinite data.".format(
                        array_name
                    )
                )
        if np.any(self.beta <= 0.0):
            raise ValueError("Reference beta must remain positive.")

        self.s_disc = float(columns["s_disc"][0])
        self.s_r = float(columns["s_r"][0])
        self.s_splice = float(columns["s_splice"][0])
        self.s_in = float(columns["s_in"][0])
        self.s_min = float(self.s[0])
        self.s_max = float(self.s[-1])

    def evaluate(self, s_value):
        s_value = clamp(s_value, self.s_min, self.s_max)
        index = int(np.searchsorted(self.s, s_value, side="left"))
        index = min(max(index, 0), len(self.s)-1)

        return {
            "s": s_value,
            "x": float(np.interp(s_value, self.s, self.x)),
            "y": float(np.interp(s_value, self.s, self.y)),
            "theta_unwrapped": float(
                np.interp(s_value, self.s, self.theta)
            ),
            "theta": normalize_angle(
                float(np.interp(s_value, self.s, self.theta))
            ),
            "kappa": float(np.interp(s_value, self.s, self.kappa)),
            "kappa_prime": float(
                np.interp(s_value, self.s, self.kappa_prime)
            ),
            "gamma": float(np.interp(s_value, self.s, self.gamma)),
            "beta": float(np.interp(s_value, self.s, self.beta)),
            "phase": self.phase[index],
        }

    def initial_guess(self, x_value, y_value, search_span):
        mask = self.s <= min(self.s_min + search_span, self.s_max)
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            return self.s_min
        distance_sq = (
            (self.x[indices] - x_value)**2
            + (self.y[indices] - y_value)**2
        )
        return float(self.s[indices[int(np.argmin(distance_sq))]])


class ActuatorProfileTable:
    """Optional MATLAB actuator-envelope profile used only for logging."""

    REQUIRED = (
        "s", "z_actuator", "d_lag_bound", "b_bar",
        "beta_actuator", "beta_terminal_actuator",
        "kappa_ref_total_cert",
    )

    def __init__(self, filename):
        filename = os.path.expanduser(filename)
        if not filename or not os.path.isfile(filename):
            raise IOError(
                "Actuator-profile CSV does not exist: {}".format(filename)
            )

        columns = {name: [] for name in self.REQUIRED}
        with open(filename, "r") as file_object:
            reader = csv.DictReader(file_object)
            if reader.fieldnames is None:
                raise ValueError("Actuator-profile CSV has no header.")
            missing = [
                name for name in self.REQUIRED
                if name not in reader.fieldnames
            ]
            if missing:
                raise ValueError(
                    "Actuator-profile CSV is missing columns: {}".format(
                        missing
                    )
                )
            for row in reader:
                for name in self.REQUIRED:
                    columns[name].append(float(row[name]))

        self.filename = filename
        for name in self.REQUIRED:
            setattr(self, name, np.asarray(columns[name], dtype=float))

        if len(self.s) < 10 or not np.all(np.diff(self.s) > 0.0):
            raise ValueError(
                "Actuator-profile coordinate must be strictly increasing."
            )

    def evaluate(self, s_value):
        if s_value < self.s[0] or s_value > self.s[-1]:
            return {
                "available": 0,
                "z_actuator": float("nan"),
                "d_lag_bound": float("nan"),
                "b_bar": float("nan"),
                "beta_actuator": float("nan"),
                "beta_terminal_actuator": float("nan"),
                "kappa_ref_total_cert": float("nan"),
            }
        result = {"available": 1}
        for name in self.REQUIRED[1:]:
            result[name] = float(
                np.interp(s_value, self.s, getattr(self, name))
            )
        return result


class Scenario1PLERealCar:
    CSV_FIELDS = [
        "wall_time_iso", "ros_time", "elapsed_time",
        "state", "fault_code", "loop_dt",
        "pose_stamp", "pose_age", "odom_stamp", "odom_age",
        "pose_x", "pose_y", "yaw_raw", "yaw_corrected",
        "odom_vx", "odom_vy", "odom_wz", "odom_speed_xy",
        "reference_phase", "event_region",
        "ref_x", "ref_y", "ref_yaw",
        "s", "e", "delta_rad", "delta_deg", "D", "eta",
        "projection_residual", "projection_iterations",
        "projection_step_abs_max",
        "gamma", "beta", "V_gamma", "V_gamma_over_beta",
        "u_ple", "a_term", "a_kappa_prime_term",
        "a_kappa_term", "b_term",
        "kappa_d", "kappa_d_prime",
        "kappa_raw", "kappa_cmd",
        "kappa_saturation_active", "certificate_violation_active",
        "v_cmd", "v_for_omega", "omega_raw", "omega_cmd",
        "omega_slew_active",
        "distance_to_discovery", "distance_to_reserve_end",
        "distance_to_splice", "distance_to_insertion",
        "inside_boundary", "raw_curvature_violation_time",
        "progress_since_track_start", "no_progress_time",
        "event_discovered", "event_spliced", "event_inserted",
        "actuator_profile_available", "z_actuator_bound",
        "d_lag_bound", "b_bar_cert", "beta_actuator_bound",
        "beta_terminal_actuator", "beta_actuator_ratio",
        "kappa_ref_total_cert",
    ]

    WAIT_POSE = "WAIT_POSE"
    WAIT_ENABLE = "WAIT_ENABLE"
    RAMP_UP = "RAMP_UP"
    TRACK = "TRACK"
    RAMP_DOWN = "RAMP_DOWN"
    FINISHED = "FINISHED"
    FAULT = "FAULT"

    def __init__(self):
        rospy.init_node("ple_scenario1_realcar", anonymous=False)

        self.pose_topic = rospy.get_param(
            "~pose_topic", "/vrpn_client_node/car4/pose"
        )
        self.odom_topic = rospy.get_param("~odom_topic", "/car4/odom")
        self.cmd_topic = rospy.get_param("~cmd_topic", "/car4/cmd_vel")
        self.reference_csv = os.path.expanduser(
            rospy.get_param(
                "~reference_csv",
                "~/ple_platform_tests/selected_candidate_reference_v2.csv",
            )
        )
        self.reference = ReferenceTable(self.reference_csv)

        self.actuator_profile_csv = os.path.expanduser(
            rospy.get_param(
                "~actuator_profile_csv",
                "~/ple_platform_tests/"
                "selected_candidate_actuator_profile_v2.csv",
            )
        )
        self.actuator_profile = None
        try:
            self.actuator_profile = ActuatorProfileTable(
                self.actuator_profile_csv
            )
        except Exception as error:
            rospy.logwarn(
                "Actuator profile disabled: %s", str(error)
            )

        # Vehicle/controller settings identified in Experiment 0.
        self.v_target = float(rospy.get_param("~v_cmd", 0.20))
        self.speed_calibration_ratio = float(
            rospy.get_param("~speed_calibration_ratio", 0.193 / 0.20)
        )
        self.kappa_cert = float(rospy.get_param("~kappa_cert", 1.45))
        self.kappa_hard = float(rospy.get_param("~kappa_hard", 1.50))
        self.omega_hard = float(rospy.get_param("~omega_hard", 0.30))
        self.omega_slew_rate = float(
            rospy.get_param("~omega_slew_rate", 2.0)
        )

        self.ramp_up_time = float(rospy.get_param("~ramp_up_time", 1.50))
        self.ramp_down_time = float(
            rospy.get_param("~ramp_down_time", 1.50)
        )
        default_stop_s = self.reference.s_max - 0.18
        self.stop_s = float(rospy.get_param("~stop_s", default_stop_s))

        # Initial-condition acceptance.
        self.initial_search_span = float(
            rospy.get_param("~initial_search_span", 0.50)
        )
        self.initial_s_max = float(rospy.get_param("~initial_s_max", 0.18))
        self.initial_e_max = float(rospy.get_param("~initial_e_max", 0.05))
        self.initial_delta_max = math.radians(
            float(rospy.get_param("~initial_delta_max_deg", 5.0))
        )
        self.initial_ratio_max = float(
            rospy.get_param("~initial_ratio_max", 0.90)
        )
        self.initial_projection_residual_max = float(
            rospy.get_param("~initial_projection_residual_max", 0.04)
        )

        # Projection settings.
        self.projection_iterations = int(
            rospy.get_param("~projection_iterations", 4)
        )
        self.projection_step_limit = float(
            rospy.get_param("~projection_step_limit", 0.12)
        )
        self.projection_residual_limit = float(
            rospy.get_param("~projection_residual_limit", 0.06)
        )

        # Runtime safety.
        self.rate_hz = float(rospy.get_param("~rate_hz", 50.0))
        self.pose_timeout = float(rospy.get_param("~pose_timeout", 0.15))
        self.max_abs_e = float(rospy.get_param("~max_abs_e", 0.22))
        self.max_abs_delta = math.radians(
            float(rospy.get_param("~max_abs_delta_deg", 25.0))
        )
        self.D_min = float(rospy.get_param("~D_min", 0.70))
        self.loop_dt_fault = float(
            rospy.get_param("~loop_dt_fault", 0.10)
        )
        self.loop_dt_fault_count_max = int(
            rospy.get_param("~loop_dt_fault_count", 2)
        )
        self.raw_kappa_violation_hold = float(
            rospy.get_param("~raw_kappa_violation_hold", 0.10)
        )
        self.no_progress_check_after = float(
            rospy.get_param("~no_progress_check_after", 3.0)
        )
        self.no_progress_min_distance = float(
            rospy.get_param("~no_progress_min_distance", 0.05)
        )

        # Match the v15 7 m x 3.5 m indoor safe region.
        self.x_min = float(rospy.get_param("~x_min", -3.00))
        self.x_max = float(rospy.get_param("~x_max", 3.80))
        self.y_min = float(rospy.get_param("~y_min", -1.45))
        self.y_max = float(rospy.get_param("~y_max", 1.45))
        self.yaw_offset = float(rospy.get_param("~yaw_offset", 0.0))
        self.auto_start = bool(rospy.get_param("~auto_start", False))

        self.lock = threading.Lock()
        self.pose = None
        self.odom = None
        self.pose_recv_wall = None
        self.odom_recv_wall = None

        self.state = self.WAIT_POSE
        self.fault_code = ""
        self.last_s = None
        self.last_omega_cmd = 0.0
        self.raw_curvature_violation_time = 0.0
        self.loop_overrun_count = 0
        self.track_start_s = None
        self.track_start_time = None
        self.discovery_announced = False
        self.reserve_announced = False
        self.splice_announced = False
        self.insertion_announced = False

        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.pose_sub = rospy.Subscriber(
            self.pose_topic, PoseStamped, self.pose_callback,
            queue_size=1, tcp_nodelay=True
        )
        self.odom_sub = rospy.Subscriber(
            self.odom_topic, Odometry, self.odom_callback,
            queue_size=1, tcp_nodelay=True
        )

        log_dir = os.path.expanduser(
            rospy.get_param("~log_dir", "~/ple_platform_tests")
        )
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = "scenario1_PLE_actuator_aware_{}.csv".format(stamp)
        self.csv_path = os.path.join(log_dir, filename)
        self.csv_file = open(self.csv_path, "w", newline="", buffering=1)
        self.csv_writer = csv.DictWriter(
            self.csv_file, fieldnames=self.CSV_FIELDS
        )
        self.csv_writer.writeheader()

        rospy.on_shutdown(self.shutdown)

        rospy.loginfo("Scenario-1 PLE real-car node initialized.")
        rospy.loginfo("Reference CSV: %s", self.reference_csv)
        if self.actuator_profile is not None:
            rospy.loginfo(
                "Actuator profile CSV: %s",
                self.actuator_profile_csv,
            )
        rospy.loginfo("Pose topic: %s", self.pose_topic)
        rospy.loginfo("Command topic: %s", self.cmd_topic)
        rospy.loginfo("Log CSV: %s", self.csv_path)

    def pose_callback(self, message):
        quaternion = message.pose.orientation
        _, _, yaw_raw = tf.transformations.euler_from_quaternion(
            (
                quaternion.x, quaternion.y,
                quaternion.z, quaternion.w,
            )
        )
        data = {
            "stamp": message.header.stamp.to_sec(),
            "x": message.pose.position.x,
            "y": message.pose.position.y,
            "yaw_raw": yaw_raw,
            "yaw": normalize_angle(yaw_raw + self.yaw_offset),
        }
        with self.lock:
            self.pose = data
            self.pose_recv_wall = time.monotonic()

    def odom_callback(self, message):
        vx = message.twist.twist.linear.x
        vy = message.twist.twist.linear.y
        data = {
            "stamp": message.header.stamp.to_sec(),
            "vx": vx,
            "vy": vy,
            "wz": message.twist.twist.angular.z,
            "speed_xy": math.hypot(vx, vy),
        }
        with self.lock:
            self.odom = data
            self.odom_recv_wall = time.monotonic()

    def snapshot(self):
        with self.lock:
            pose = dict(self.pose) if self.pose is not None else None
            odom = dict(self.odom) if self.odom is not None else None
            pose_recv = self.pose_recv_wall
            odom_recv = self.odom_recv_wall

        now = time.monotonic()
        pose_age = float("inf") if pose_recv is None else now-pose_recv
        odom_age = float("inf") if odom_recv is None else now-odom_recv
        return pose, odom, pose_age, odom_age

    def project_to_reference(self, pose):
        if self.last_s is None:
            s_value = self.reference.initial_guess(
                pose["x"], pose["y"], self.initial_search_span
            )
        else:
            s_value = self.last_s

        step_abs_max = 0.0
        iterations_used = 0
        for iteration in range(self.projection_iterations):
            ref = self.reference.evaluate(s_value)
            tx = math.cos(ref["theta_unwrapped"])
            ty = math.sin(ref["theta_unwrapped"])
            nx = -ty
            ny = tx
            dx = pose["x"] - ref["x"]
            dy = pose["y"] - ref["y"]
            residual = dx*tx + dy*ty
            e_value = dx*nx + dy*ny
            D_value = 1.0 - ref["kappa"]*e_value

            if abs(D_value) < 0.20:
                break

            step = clamp(
                residual/D_value,
                -self.projection_step_limit,
                self.projection_step_limit,
            )
            step_abs_max = max(step_abs_max, abs(step))
            s_value = clamp(
                s_value + step,
                self.reference.s_min,
                self.reference.s_max,
            )
            iterations_used = iteration + 1
            if abs(step) < 1.0e-6:
                break

        ref = self.reference.evaluate(s_value)
        tx = math.cos(ref["theta_unwrapped"])
        ty = math.sin(ref["theta_unwrapped"])
        nx = -ty
        ny = tx
        dx = pose["x"] - ref["x"]
        dy = pose["y"] - ref["y"]

        residual = dx*tx + dy*ty
        e_value = dx*nx + dy*ny
        delta_value = normalize_angle(pose["yaw"] - ref["theta"])
        D_value = 1.0 - ref["kappa"]*e_value
        eta_value = D_value*math.tan(delta_value)

        self.last_s = s_value
        return {
            "s": s_value,
            "e": e_value,
            "delta": delta_value,
            "D": D_value,
            "eta": eta_value,
            "projection_residual": residual,
            "projection_iterations": iterations_used,
            "projection_step_abs_max": step_abs_max,
            "ref": ref,
        }

    def ple_quantities(self, frenet):
        ref = frenet["ref"]
        e_value = frenet["e"]
        eta_value = frenet["eta"]
        D_value = frenet["D"]
        gamma = ref["gamma"]
        beta = ref["beta"]
        kappa_d = ref["kappa"]
        kappa_prime = ref["kappa_prime"]

        V_gamma = (
            gamma**3*e_value*e_value
            + 2.0*gamma**2*e_value*eta_value
            + 2.0*gamma*eta_value*eta_value
        )
        u_ple = -gamma**2*e_value - 2.0*gamma*eta_value

        a_kappa_prime_term = kappa_prime*e_value*eta_value
        a_kappa_term = kappa_d*(D_value*D_value + 2.0*eta_value*eta_value)
        a_term = -(a_kappa_prime_term + a_kappa_term)/D_value
        b_term = (
            (D_value*D_value + eta_value*eta_value)**1.5
            / D_value
        )
        kappa_raw = (u_ple-a_term)/b_term
        kappa_cmd = clamp(
            kappa_raw, -self.kappa_hard, self.kappa_hard
        )

        return {
            "gamma": gamma,
            "beta": beta,
            "V_gamma": V_gamma,
            "u_ple": u_ple,
            "a_term": a_term,
            "a_kappa_prime_term": a_kappa_prime_term,
            "a_kappa_term": a_kappa_term,
            "b_term": b_term,
            "kappa_raw": kappa_raw,
            "kappa_cmd": kappa_cmd,
            "kappa_saturation_active": int(
                abs(kappa_raw) > self.kappa_hard + 1.0e-12
            ),
            "certificate_violation_active": int(V_gamma > beta),
        }

    def event_region(self, s_value):
        if s_value < self.reference.s_disc:
            return "pre_discovery"
        if s_value < self.reference.s_r:
            return "P0_compute_reserve"
        if s_value < self.reference.s_splice:
            return "P1_gamma_ramp"
        if s_value < self.reference.s_in:
            return "P2_bridge"
        return "terminal_post_insertion"

    def announce_event_progress(self, s_value):
        if (
            not self.discovery_announced
            and s_value >= self.reference.s_disc
        ):
            self.discovery_announced = True
            rospy.logwarn(
                "TERMINAL REFERENCE DISCOVERED at s=%.3f m. "
                "Entering P0 compute reserve.",
                s_value,
            )
        if (
            not self.reserve_announced
            and s_value >= self.reference.s_r
        ):
            self.reserve_announced = True
            rospy.loginfo(
                "P0 reserve completed at s=%.3f m. "
                "Starting gamma ramp.",
                s_value,
            )
        if (
            not self.splice_announced
            and s_value >= self.reference.s_splice
        ):
            self.splice_announced = True
            rospy.loginfo(
                "Bridge splice reached at s=%.3f m.", s_value
            )
        if (
            not self.insertion_announced
            and s_value >= self.reference.s_in
        ):
            self.insertion_announced = True
            rospy.logwarn(
                "TERMINAL INSERTION reached at s=%.3f m.", s_value
            )

    def inside_boundary(self, pose):
        return (
            self.x_min <= pose["x"] <= self.x_max
            and self.y_min <= pose["y"] <= self.y_max
        )

    def publish_command(self, v_cmd, omega_raw, loop_dt):
        omega_raw = clamp(
            omega_raw, -self.omega_hard, self.omega_hard
        )
        max_change = self.omega_slew_rate*max(loop_dt, 0.0)
        omega_cmd = self.last_omega_cmd + clamp(
            omega_raw-self.last_omega_cmd,
            -max_change, max_change
        )
        omega_cmd = clamp(
            omega_cmd, -self.omega_hard, self.omega_hard
        )

        message = Twist()
        message.linear.x = v_cmd
        message.angular.z = omega_cmd
        self.cmd_pub.publish(message)

        slew_active = int(abs(omega_cmd-omega_raw) > 1.0e-9)
        self.last_omega_cmd = omega_cmd
        return omega_cmd, slew_active

    def publish_zero(self, repeats=15):
        message = Twist()
        for _ in range(repeats):
            self.cmd_pub.publish(message)
            rospy.sleep(0.02)
        self.last_omega_cmd = 0.0

    def set_fault(self, code):
        self.state = self.FAULT
        self.fault_code = code
        rospy.logerr("FAULT: %s", code)
        self.publish_zero()

    def initial_condition_errors(self, frenet, ple):
        errors = []
        if frenet["s"] > self.initial_s_max:
            errors.append(
                "initial s={:.3f} > {:.3f} m".format(
                    frenet["s"], self.initial_s_max
                )
            )
        if abs(frenet["e"]) > self.initial_e_max:
            errors.append(
                "|e0|={:.3f} > {:.3f} m".format(
                    abs(frenet["e"]), self.initial_e_max
                )
            )
        if abs(frenet["delta"]) > self.initial_delta_max:
            errors.append(
                "|delta0|={:.2f} > {:.2f} deg".format(
                    abs(math.degrees(frenet["delta"])),
                    math.degrees(self.initial_delta_max),
                )
            )
        if frenet["D"] < self.D_min:
            errors.append(
                "D0={:.3f} < {:.3f}".format(frenet["D"], self.D_min)
            )
        if abs(frenet["projection_residual"]) > (
            self.initial_projection_residual_max
        ):
            errors.append(
                "initial projection residual={:.3f} m is too large".format(
                    frenet["projection_residual"]
                )
            )
        ratio = ple["V_gamma"]/ple["beta"]
        if ratio > self.initial_ratio_max:
            errors.append(
                "initial V/beta={:.3f} > {:.3f}".format(
                    ratio, self.initial_ratio_max
                )
            )
        return errors

    def wait_for_valid_pose(self):
        self.state = self.WAIT_POSE
        rospy.loginfo("Waiting for motion-capture pose...")
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            pose, _, pose_age, _ = self.snapshot()
            if pose is not None and pose_age <= self.pose_timeout:
                return pose
            rate.sleep()
        return None

    def print_reference_information(self, pose, frenet, ple):
        start = self.reference.evaluate(self.reference.s_min)
        finish = self.reference.evaluate(self.reference.s_max)
        rospy.loginfo("--------------------------------------------------")
        rospy.loginfo(
            "Reference length: %.3f m", self.reference.s_max
        )
        rospy.loginfo(
            "Reference start: (%.3f, %.3f), yaw %.2f deg",
            start["x"], start["y"], math.degrees(start["theta"])
        )
        rospy.loginfo(
            "Reference end:   (%.3f, %.3f), yaw %.2f deg",
            finish["x"], finish["y"], math.degrees(finish["theta"])
        )
        rospy.loginfo(
            "Event s: disc %.3f, reserve %.3f, splice %.3f, insertion %.3f",
            self.reference.s_disc,
            self.reference.s_r,
            self.reference.s_splice,
            self.reference.s_in,
        )
        rospy.loginfo(
            "Reference maxima: |kappa| %.3f 1/m, |kappa'| %.3f 1/m^2",
            float(np.max(np.abs(self.reference.kappa))),
            float(np.max(np.abs(self.reference.kappa_prime))),
        )
        rospy.loginfo(
            "Current pose: (%.3f, %.3f), yaw %.2f deg",
            pose["x"], pose["y"], math.degrees(pose["yaw"])
        )
        rospy.loginfo(
            "Initial Frenet: s=%.3f, e=%+.3f m, delta=%+.2f deg, D=%.3f",
            frenet["s"], frenet["e"],
            math.degrees(frenet["delta"]), frenet["D"]
        )
        rospy.loginfo(
            "Initial schedule: phase=%s, gamma=%.3f, beta=%.6g, V/beta=%.3f",
            frenet["ref"]["phase"], ple["gamma"], ple["beta"],
            ple["V_gamma"]/ple["beta"],
        )
        rospy.loginfo(
            "Initial command: kappa_raw=%+.3f 1/m",
            ple["kappa_raw"]
        )
        rospy.loginfo("--------------------------------------------------")

    def confirm_start(self):
        if self.auto_start:
            return True
        try:
            answer = input(
                "Type RUN and press Enter to start Scenario 1: "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer == "RUN"

    def log_row(
        self, elapsed, loop_dt, pose, odom, pose_age, odom_age,
        frenet, ple, v_cmd, v_for_omega, omega_raw, omega_cmd,
        slew_active, progress, no_progress_time
    ):
        odom = odom or {}
        ref = frenet["ref"]
        s_value = frenet["s"]
        if self.actuator_profile is None:
            actuator = {"available": 0}
        else:
            actuator = self.actuator_profile.evaluate(s_value)

        beta_actuator = actuator.get(
            "beta_actuator", float("nan")
        )
        beta_terminal_actuator = actuator.get(
            "beta_terminal_actuator", float("nan")
        )
        if (
            math.isfinite(beta_actuator)
            and math.isfinite(beta_terminal_actuator)
            and beta_terminal_actuator > 0.0
        ):
            beta_actuator_ratio = (
                beta_actuator / beta_terminal_actuator
            )
        else:
            beta_actuator_ratio = float("nan")

        row = {
            "wall_time_iso": datetime.now().isoformat(
                timespec="milliseconds"
            ),
            "ros_time": rospy.Time.now().to_sec(),
            "elapsed_time": elapsed,
            "state": self.state,
            "fault_code": self.fault_code,
            "loop_dt": loop_dt,
            "pose_stamp": pose["stamp"],
            "pose_age": pose_age,
            "odom_stamp": odom.get("stamp", ""),
            "odom_age": odom_age,
            "pose_x": pose["x"],
            "pose_y": pose["y"],
            "yaw_raw": pose["yaw_raw"],
            "yaw_corrected": pose["yaw"],
            "odom_vx": odom.get("vx", ""),
            "odom_vy": odom.get("vy", ""),
            "odom_wz": odom.get("wz", ""),
            "odom_speed_xy": odom.get("speed_xy", ""),
            "reference_phase": ref["phase"],
            "event_region": self.event_region(s_value),
            "ref_x": ref["x"],
            "ref_y": ref["y"],
            "ref_yaw": ref["theta"],
            "s": s_value,
            "e": frenet["e"],
            "delta_rad": frenet["delta"],
            "delta_deg": math.degrees(frenet["delta"]),
            "D": frenet["D"],
            "eta": frenet["eta"],
            "projection_residual": frenet["projection_residual"],
            "projection_iterations": frenet["projection_iterations"],
            "projection_step_abs_max": frenet[
                "projection_step_abs_max"
            ],
            "gamma": ple["gamma"],
            "beta": ple["beta"],
            "V_gamma": ple["V_gamma"],
            "V_gamma_over_beta": ple["V_gamma"]/ple["beta"],
            "u_ple": ple["u_ple"],
            "a_term": ple["a_term"],
            "a_kappa_prime_term": ple["a_kappa_prime_term"],
            "a_kappa_term": ple["a_kappa_term"],
            "b_term": ple["b_term"],
            "kappa_d": ref["kappa"],
            "kappa_d_prime": ref["kappa_prime"],
            "kappa_raw": ple["kappa_raw"],
            "kappa_cmd": ple["kappa_cmd"],
            "kappa_saturation_active": ple[
                "kappa_saturation_active"
            ],
            "certificate_violation_active": ple[
                "certificate_violation_active"
            ],
            "v_cmd": v_cmd,
            "v_for_omega": v_for_omega,
            "omega_raw": omega_raw,
            "omega_cmd": omega_cmd,
            "omega_slew_active": slew_active,
            "distance_to_discovery": self.reference.s_disc-s_value,
            "distance_to_reserve_end": self.reference.s_r-s_value,
            "distance_to_splice": self.reference.s_splice-s_value,
            "distance_to_insertion": self.reference.s_in-s_value,
            "inside_boundary": int(self.inside_boundary(pose)),
            "raw_curvature_violation_time": (
                self.raw_curvature_violation_time
            ),
            "progress_since_track_start": progress,
            "no_progress_time": no_progress_time,
            "event_discovered": int(
                s_value >= self.reference.s_disc
            ),
            "event_spliced": int(
                s_value >= self.reference.s_splice
            ),
            "event_inserted": int(
                s_value >= self.reference.s_in
            ),
            "actuator_profile_available": actuator.get(
                "available", 0
            ),
            "z_actuator_bound": actuator.get(
                "z_actuator", float("nan")
            ),
            "d_lag_bound": actuator.get(
                "d_lag_bound", float("nan")
            ),
            "b_bar_cert": actuator.get(
                "b_bar", float("nan")
            ),
            "beta_actuator_bound": beta_actuator,
            "beta_terminal_actuator": beta_terminal_actuator,
            "beta_actuator_ratio": beta_actuator_ratio,
            "kappa_ref_total_cert": actuator.get(
                "kappa_ref_total_cert", float("nan")
            ),
        }
        self.csv_writer.writerow(row)

    def run(self):
        pose = self.wait_for_valid_pose()
        if pose is None:
            return

        frenet0 = self.project_to_reference(pose)
        ple0 = self.ple_quantities(frenet0)
        self.print_reference_information(pose, frenet0, ple0)

        errors = self.initial_condition_errors(frenet0, ple0)
        if errors:
            for error in errors:
                rospy.logerr("Initial-condition rejection: %s", error)
            self.set_fault("INITIAL_CONDITION_REJECTED")
            return
        if not self.inside_boundary(pose):
            self.set_fault("INITIAL_BOUNDARY_REJECTED")
            return

        self.state = self.WAIT_ENABLE
        self.publish_zero(repeats=5)
        if not self.confirm_start():
            rospy.logwarn("Experiment cancelled.")
            self.publish_zero()
            return

        start_wall = time.monotonic()
        state_start = start_wall
        last_loop = start_wall
        self.state = self.RAMP_UP
        rate = rospy.Rate(self.rate_hz)

        while not rospy.is_shutdown():
            now = time.monotonic()
            elapsed = now-start_wall
            loop_dt = now-last_loop
            last_loop = now

            pose, odom, pose_age, odom_age = self.snapshot()
            if pose is None or pose_age > self.pose_timeout:
                self.set_fault("POSE_TIMEOUT")
                break
            if not self.inside_boundary(pose):
                self.set_fault("BOUNDARY_VIOLATION")
                break

            if loop_dt > self.loop_dt_fault:
                self.loop_overrun_count += 1
            else:
                self.loop_overrun_count = 0
            if self.loop_overrun_count >= self.loop_dt_fault_count_max:
                self.set_fault("CONTROL_LOOP_OVERRUN")
                break

            frenet = self.project_to_reference(pose)
            ple = self.ple_quantities(frenet)
            self.announce_event_progress(frenet["s"])

            if abs(frenet["projection_residual"]) > (
                self.projection_residual_limit
            ):
                self.set_fault("PROJECTION_RESIDUAL_LIMIT")
                break
            if frenet["D"] < self.D_min:
                self.set_fault("FRENET_D_LIMIT")
                break
            if abs(frenet["e"]) > self.max_abs_e:
                self.set_fault("LATERAL_ERROR_LIMIT")
                break
            if abs(frenet["delta"]) > self.max_abs_delta:
                self.set_fault("HEADING_ERROR_LIMIT")
                break

            if abs(ple["kappa_raw"]) > self.kappa_hard:
                self.raw_curvature_violation_time += loop_dt
            else:
                self.raw_curvature_violation_time = 0.0
            if self.raw_curvature_violation_time >= (
                self.raw_kappa_violation_hold
            ):
                self.set_fault("PERSISTENT_RAW_CURVATURE_LIMIT")
                break

            state_elapsed = now-state_start
            if self.state == self.RAMP_UP:
                fraction = state_elapsed/max(self.ramp_up_time, 1.0e-6)
                v_cmd = self.v_target*smoothstep01(fraction)
                if fraction >= 1.0:
                    self.state = self.TRACK
                    state_start = now
                    self.track_start_s = frenet["s"]
                    self.track_start_time = now
                    v_cmd = self.v_target
            elif self.state == self.TRACK:
                v_cmd = self.v_target
                if frenet["s"] >= self.stop_s:
                    self.state = self.RAMP_DOWN
                    state_start = now
            elif self.state == self.RAMP_DOWN:
                fraction = state_elapsed/max(self.ramp_down_time, 1.0e-6)
                v_cmd = self.v_target*(1.0-smoothstep01(fraction))
                if fraction >= 1.0:
                    self.state = self.FINISHED
                    v_cmd = 0.0
            elif self.state == self.FINISHED:
                self.publish_zero()
                rospy.loginfo("Scenario 1 completed.")
                rospy.loginfo("CSV saved to: %s", self.csv_path)
                break
            else:
                v_cmd = 0.0

            progress = 0.0
            no_progress_time = 0.0
            if self.track_start_s is not None:
                progress = frenet["s"]-self.track_start_s
                no_progress_time = now-self.track_start_time
            if (
                self.state == self.TRACK
                and no_progress_time > self.no_progress_check_after
                and progress < self.no_progress_min_distance
            ):
                self.set_fault("NO_FORWARD_PROGRESS")
                break

            v_for_omega = (
                self.speed_calibration_ratio*max(v_cmd, 0.0)
            )
            omega_raw = v_for_omega*ple["kappa_cmd"]
            omega_cmd, slew_active = self.publish_command(
                v_cmd, omega_raw, loop_dt
            )

            self.log_row(
                elapsed, loop_dt, pose, odom, pose_age, odom_age,
                frenet, ple, v_cmd, v_for_omega, omega_raw, omega_cmd,
                slew_active, progress, no_progress_time
            )

            if int(elapsed*2.0) != int((elapsed-loop_dt)*2.0):
                rospy.loginfo(
                    "%s | %s | s=%.2f | e=%+.3f | delta=%+.2f deg | "
                    "gamma=%.2f | kd=%+.3f | kc=%+.3f | "
                    "omega=%+.3f | V/beta=%.3f",
                    self.state,
                    self.event_region(frenet["s"]),
                    frenet["s"],
                    frenet["e"],
                    math.degrees(frenet["delta"]),
                    ple["gamma"],
                    frenet["ref"]["kappa"],
                    ple["kappa_cmd"],
                    omega_cmd,
                    ple["V_gamma"]/ple["beta"],
                )
            rate.sleep()

    def shutdown(self):
        try:
            self.publish_zero(repeats=10)
        except Exception:
            pass
        if not self.csv_file.closed:
            self.csv_file.close()
            rospy.loginfo("CSV closed: %s", self.csv_path)


if __name__ == "__main__":
    try:
        Scenario1PLERealCar().run()
    except rospy.ROSInterruptException:
        pass
