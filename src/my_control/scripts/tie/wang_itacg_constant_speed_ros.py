#!/usr/bin/env python3
# coding: utf-8

import math
import csv
import os
from datetime import datetime

import rospy
import tf

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64


class WangITACGGuidance:
    def __init__(self):
        rospy.init_node("wang_itacg_guidance", anonymous=True)

        # ============================================================
        # 1. Gate task
        # ============================================================
        self.target_x = -1.5
        self.target_y = 1.5
        self.lambda_desired = math.radians(135.0)

        # Desired first crossing time measured from node mission start.
        self.desired_total_time = 20.0

        # Constant-speed common subset used for Wang comparison.
        self.cruise_speed = 0.23

        # Short startup speed ramp.  Wang initialization is performed
        # AFTER this ramp using the measured pose.
        self.accel_duration = 0.50

        # Li-style run-in: place car at (2.0,2.0), start clock when crossing y=1.5
        self.start_x = 2.0
        self.start_y = 1.5
        self.start_yaw = math.radians(-90.0)
        self.runin_start_y = 2.0

        # ============================================================
        # 2. Wang controller parameters
        # ============================================================
        self.n = 3.0
        self.h1 = 0.25
        self.p_exp = 0.8
        self.mu = 0.9

        # "paper"            -> printed Eq. (79), scale = 1
        # "proof_consistent" -> scale = n(n+1) = 12 for n=3
        self.bias_mode = "paper"

        # Practical numerical settings for K(e).
        self.use_smooth_sign = False
        self.sign_smoothing_a = 0.01
        self.e_deadband = 0.005  # [s]

        # ============================================================
        # 3. Physical command limits
        # ============================================================
        self.max_angular = 3.0       # [rad/s]
        self.max_yaw_accel = 2.0     # [rad/s^2]
        self.last_cmd_omega = 0.0

        # Numerical terminal protection.
        self.R_floor = 1.0e-6
        self.varpi_floor = 1.0e-6
        # Numerical singularity guard only. NOT a gate-passing stop condition.
        self.terminal_guard_R = 0.08
        self.terminal_guard_angle = math.radians(2.0)

        # Terminal angle-lock protection (avoid varpi singularity)
        # Similar to itacg_disturbance.py: when the remaining geometry
        # enters the terminal region, freeze desired heading and fly straight.
        self.terminal_lock_sigma = math.radians(5.0)
        self.terminal_lock_lambda_error = math.radians(3.0)
        self.terminal_lock_R = 0.6
        self.terminal_lock_tgo = 0.8
        self.terminal_lock_varpi = math.radians(5.0)
        self.locked_yaw = None
        self.terminal_omega_gain = 1.2

        # ============================================================
        # 4. Optional command-channel disturbance
        # ============================================================
        self.inject_disturbance = True
        self.dist_start_time = 5.0
        self.dist_duration = 1.0
        self.dist_omega_val = 0.5

        # ============================================================
        # 5. Runtime states
        # ============================================================
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.actual_speed = 0.0

        self.pose_ready = False
        self.odom_ready = False

        self.STATE_RUNIN = 1
        self.STATE_GUIDANCE = 2
        self.STATE_LOCK = 3
        self.STATE_TERMINAL = 4
        self.state = self.STATE_RUNIN

        self.node_start = None
        self.mission_start = None
        self.guidance_start = None

        self.prev_start_y = None
        self.prev_start_x = None
        self.prev_start_time = None
        self.last_time_sec = None

        # Wang initialization values.
        self.T_guidance = None
        self.e0 = None
        self.K_coeff = None
        self.bias_scale = None
        self.guidance_initialized = False

        # First-crossing interpolation memory.
        self.prev_gate_d = None
        self.prev_guidance_t = None
        self.prev_mission_t = None
        self.prev_x = None
        self.prev_y = None
        self.prev_yaw = None

        # Crossing result storage
        self.crossing_finished = False
        self.cross_time = None
        self.cross_eT = None
        self.cross_ep = None
        self.cross_etheta = None

        # ============================================================
        # 6. CSV logger
        # ============================================================
        log_dir = os.path.expanduser("~")
        filename = datetime.now().strftime(
            "wang_itacg_data_%Y%m%d_%H%M%S.csv"
        )
        self.csv_path = os.path.join(log_dir, filename)
        self.csv_file = open(self.csv_path, "w")
        self.csv_writer = csv.writer(self.csv_file)

        self.csv_keys = [
            "mission_time",
            "guidance_time",
            "relative_dist",
            "tgo_desired",
            "tgo_model",
            "e_tgo",
            "speed_actual",
            "speed_desired",
            "sigma_actual",
            "lambda_actual",
            "lambda_error",
            "varpi",
            "K_e",
            "A_iac",
            "A_bias",
            "A_itac",
            "omega_raw",
            "omega_cmd",
            "yaw_actual",
            "pos_x",
            "pos_y",
        ]

        headers = []
        for key in self.csv_keys:
            if key in [
                "sigma_actual",
                "lambda_actual",
                "lambda_error",
                "varpi",
                "yaw_actual",
            ]:
                headers.append(key + "_deg")
            else:
                headers.append(key)
        self.csv_writer.writerow(headers)
        rospy.on_shutdown(self.shutdown_hook)

        # ============================================================
        # 7. ROS interfaces
        # ============================================================
        rospy.Subscriber(
            "/vrpn_client_node/car1/pose",
            PoseStamped,
            self.pose_callback,
        )
        rospy.Subscriber(
            "/car1/odom",
            Odometry,
            self.odom_callback,
        )
        self.cmd_pub = rospy.Publisher(
            "/car1/cmd_vel", Twist, queue_size=10
        )

        pub_topics = [
            "relative_dist",
            "tgo_desired",
            "tgo_model",
            "e_tgo",
            "speed_desired",
            "speed_actual",
            "sigma_actual",
            "lambda_actual",
            "lambda_error",
            "varpi",
            "K_e",
            "A_iac",
            "A_bias",
            "A_itac",
            "omega_raw",
            "omega_cmd",
            "yaw_actual",
        ]
        self.pubs = {
            name: rospy.Publisher(
                "/wang_itacg/{}".format(name),
                Float64,
                queue_size=10,
            )
            for name in pub_topics
        }

        self.rate = rospy.Rate(50)

        rospy.loginfo(
            "Wang ITACG node ready | bias_mode=%s | V=%.3f m/s | "
            "T_total=%.2f s",
            self.bias_mode,
            self.cruise_speed,
            self.desired_total_time,
        )

    # ================================================================
    # Basic utilities
    # ================================================================
    def shutdown_hook(self):
        try:
            self.publish_cmd_direct(0.0, 0.0)
        except Exception:
            pass

        if hasattr(self, "csv_file") and not self.csv_file.closed:
            self.csv_file.close()
            rospy.loginfo(
                "CSV saved to: %s",
                self.csv_path,
            )

    @staticmethod
    def sat(x, lo, hi):
        return max(lo, min(hi, x))

    @staticmethod
    def normalize_angle(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def slew_limit(self, x, x_last, rate, dt):
        delta = self.sat(
            x - x_last,
            -rate * dt,
            rate * dt,
        )
        return x_last + delta

    @staticmethod
    def sign_nonzero(x):
        return 1.0 if x >= 0.0 else -1.0

    # ================================================================
    # ROS callbacks
    # ================================================================
    def pose_callback(self, msg):
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y

        q = msg.pose.orientation
        _, _, self.current_yaw = tf.transformations.euler_from_quaternion(
            (q.x, q.y, q.z, q.w)
        )
        self.pose_ready = True

    def odom_callback(self, msg):
        self.actual_speed = math.hypot(
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
        )
        self.odom_ready = True

    # ================================================================
    # Geometry
    # ================================================================
    def get_geometry(self):
        dx = self.target_x - self.current_x
        dy = self.target_y - self.current_y

        R = math.hypot(dx, dy)
        los = math.atan2(dy, dx)

        sigma = self.normalize_angle(self.current_yaw - los)
        lambda_e = self.normalize_angle(los - self.lambda_desired)

        # Wang Eq. (32), n = 3 for the first illustrative law.
        varpi = (sigma + self.n * lambda_e) / (self.n + 1.0)

        return R, los, sigma, lambda_e, varpi

    def gate_signed_distance(self, x, y):
        return (
            math.cos(self.lambda_desired) * (x - self.target_x)
            + math.sin(self.lambda_desired) * (y - self.target_y)
        )

    # ================================================================
    # Wang initialization
    # ================================================================
    def initialize_wang(self):
        self.T_guidance = (
            self.desired_total_time - self.accel_duration
        )
        if self.T_guidance <= 0.0:
            raise RuntimeError("T_guidance must be positive.")

        R0, los0, sigma0, lambda_e0, varpi0 = self.get_geometry()

        if abs(lambda_e0) <= 1.0e-8:
            raise RuntimeError(
                "lambda_e0 is too small for Wang Theorem-2 ratio."
            )

        ratio0 = sigma0 / lambda_e0

        if not (0.0 < ratio0 <= self.n):
            raise RuntimeError(
                "Wang initial condition violated: "
                "0 < sigma0/lambda_e0 <= n. "
                "Current ratio = {:.6f}".format(ratio0)
            )

        if not (1.0 / 6.0 < self.h1 < 2.0 / 7.0):
            raise RuntimeError(
                "h1 must satisfy 1/6 < h1 < 2/7."
            )

        g10 = 1.0 + self.h1 * varpi0 * varpi0

        # Wang Eq. (58): exact constant-speed t_go.
        tgo0 = R0 / self.cruise_speed * g10

        self.e0 = self.T_guidance - tgo0

        # For the current implementation, use the delayed-impact case e0>0.
        if self.e0 <= 0.0:
            raise RuntimeError(
                "Wang Eq. (71) test requires e0>0 in this node. "
                "Current e0 = {:.6f} s. "
                "Adjust cruise_speed / T / initial geometry.".format(
                    self.e0
                )
            )

        self.K_coeff = (
            abs(self.e0) ** (1.0 - self.p_exp)
            / (
                self.mu
                * self.T_guidance
                * (1.0 - self.p_exp)
            )
        )

        if self.bias_mode == "paper":
            self.bias_scale = 1.0
        elif self.bias_mode == "proof_consistent":
            self.bias_scale = self.n * (self.n + 1.0)
        else:
            raise RuntimeError(
                "Unknown bias_mode: {}".format(self.bias_mode)
            )

        self.guidance_initialized = True

        self.prev_gate_d = self.gate_signed_distance(
            self.current_x,
            self.current_y,
        )
        self.prev_guidance_t = 0.0
        self.prev_mission_t = self.accel_duration
        self.prev_x = self.current_x
        self.prev_y = self.current_y
        self.prev_yaw = self.current_yaw

        rospy.loginfo("========== Wang initialization ==========")
        rospy.loginfo("R0                = %.6f m", R0)
        rospy.loginfo(
            "sigma0            = %+.6f deg",
            math.degrees(sigma0),
        )
        rospy.loginfo(
            "lambda_e0         = %+.6f deg",
            math.degrees(lambda_e0),
        )
        rospy.loginfo(
            "varpi0            = %+.6f deg",
            math.degrees(varpi0),
        )
        rospy.loginfo(
            "sigma0/lambda_e0  = %.6f",
            ratio0,
        )
        rospy.loginfo("h1                = %.6f", self.h1)
        rospy.loginfo("T_guidance        = %.6f s", self.T_guidance)
        rospy.loginfo("tgo0              = %.6f s", tgo0)
        rospy.loginfo("e0                = %+.6f s", self.e0)
        rospy.loginfo(
            "mu*T_guidance     = %.6f s",
            self.mu * self.T_guidance,
        )
        rospy.loginfo(
            "bias_mode         = %s",
            self.bias_mode,
        )
        rospy.loginfo(
            "bias_scale        = %.6f",
            self.bias_scale,
        )
        rospy.loginfo("========================================")

    # ================================================================
    # Wang guidance law
    # ================================================================
    def calc_K(self, e):
        # Small numerical deadband prevents a residual floating-point
        # e from being divided by R*varpi close to the terminal point.
        if abs(e) <= self.e_deadband:
            return 0.0

        if self.use_smooth_sign:
            sgn_e = e / (abs(e) + self.sign_smoothing_a)
        else:
            sgn_e = self.sign_nonzero(e)

        return (
            self.K_coeff
            * sgn_e
            * (abs(e) ** self.p_exp)
        )

    def calc_wang_command(self, guidance_t):
        R, los, sigma, lambda_e, varpi = self.get_geometry()

        g1 = 1.0 + self.h1 * varpi * varpi

        # Wang Eq. (58).
        tgo_model = R / self.cruise_speed * g1

        tgo_desired = self.T_guidance - guidance_t

        # Wang Eq. (68).
        e_tgo = tgo_desired - tgo_model

        # Wang Eq. (71).
        K_e = self.calc_K(e_tgo)

        terminal_guard = (
            R <= self.terminal_guard_R
            and abs(sigma) <= self.terminal_guard_angle
            and abs(lambda_e) <= self.terminal_guard_angle
        )

        if terminal_guard:
            A_iac = 0.0
            A_bias = 0.0
            A_itac = 0.0
            omega_raw = 0.0
        else:
            if R <= self.R_floor:
                raise RuntimeError(
                    "R too small outside terminal guard."
                )

            if abs(varpi) <= self.varpi_floor:
                raise RuntimeError(
                    "|varpi| too small outside terminal guard."
                )

            # Numerically stable small-angle trigonometric terms.
            if abs(sigma) < 1.0e-5:
                cos_sigma = 1.0 - 0.5 * sigma * sigma
                cos_minus_one = -0.5 * sigma * sigma
                sin_sigma = sigma
            else:
                cos_sigma = math.cos(sigma)
                cos_minus_one = cos_sigma - 1.0
                sin_sigma = math.sin(sigma)

            # Wang Eq. (57): first illustrative IACG.
            A_iac = (
                2.0
                * self.cruise_speed
                * self.cruise_speed
                / R
                * (
                    cos_minus_one / (self.h1 * varpi)
                    + varpi * cos_sigma
                    + sin_sigma
                )
            )

            # Wang Eq. (79) structure.
            #
            # bias_scale = 1:
            #     printed Eq. (79)
            #
            # bias_scale = 12 for n=3:
            #     proof-consistent version verified against Eq. (76)
            A_bias = (
                self.bias_scale
                * self.cruise_speed
                * self.cruise_speed
                / (6.0 * self.h1 * R * varpi)
                * K_e
            )

            A_itac = A_iac + A_bias

            # Missile lateral acceleration -> robot yaw rate.
            omega_raw = A_itac / self.cruise_speed

        data = {
            "R": R,
            "los": los,
            "sigma": sigma,
            "lambda_e": lambda_e,
            "varpi": varpi,
            "tgo_desired": tgo_desired,
            "tgo_model": tgo_model,
            "e_tgo": e_tgo,
            "K_e": K_e,
            "A_iac": A_iac,
            "A_bias": A_bias,
            "A_itac": A_itac,
            "omega_raw": omega_raw,
        }
        return omega_raw, data

    # ================================================================
    # Terminal heading lock protection
    # ================================================================
    def check_terminal_lock(self, data):
        R = data["R"]
        sigma = data["sigma"]
        lambda_e = data["lambda_e"]
        varpi = data["varpi"]
        tgo = data["tgo_model"]

        # Late terminal protection only.  It must NOT replace the
        # physical gate crossing.  The vehicle continues to move and
        # first-crossing evaluation remains active.
        trigger = (
            R <= self.terminal_lock_R
            and abs(varpi) <= self.terminal_lock_varpi
        ) or (
            tgo <= self.terminal_lock_tgo
            and R <= 0.2
        )

        if trigger:
            self.locked_yaw = self.lambda_desired
            rospy.loginfo(
                "Terminal angle lock activated: R=%.3f sigma=%.2f deg",
                R, math.degrees(sigma)
            )
            return True
        return False

    def calc_terminal_lock_command(self):
        if self.locked_yaw is None:
            self.locked_yaw = self.lambda_desired

        yaw_error = self.normalize_angle(
            self.locked_yaw - self.current_yaw
        )

        omega = self.terminal_omega_gain * yaw_error
        return omega

    # ================================================================
    # Command / logging
    # ================================================================
    def publish_cmd_direct(self, v, omega):
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = self.sat(
            omega,
            -self.max_angular,
            self.max_angular,
        )
        self.cmd_pub.publish(cmd)

    def publish_cmd(self, v, omega, dt):
        omega_limited = self.slew_limit(
            omega,
            self.last_cmd_omega,
            self.max_yaw_accel,
            dt,
        )

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = self.sat(
            omega_limited,
            -self.max_angular,
            self.max_angular,
        )

        self.last_cmd_omega = cmd.angular.z
        self.cmd_pub.publish(cmd)

        return cmd.angular.z

    def publish_telemetry(self, telemetry):
        for key, val in telemetry.items():
            if key not in self.pubs:
                continue

            if key in [
                "sigma_actual",
                "lambda_actual",
                "lambda_error",
                "varpi",
                "yaw_actual",
            ]:
                self.pubs[key].publish(math.degrees(val))
            else:
                self.pubs[key].publish(val)

    def record_csv(self, telemetry):
        row = []
        for key in self.csv_keys:
            val = telemetry.get(key, 0.0)

            if key in [
                "sigma_actual",
                "lambda_actual",
                "lambda_error",
                "varpi",
                "yaw_actual",
            ]:
                val = math.degrees(val)

            row.append(val)

        self.csv_writer.writerow(row)

    # ================================================================
    # First gate-crossing evaluator
    # ================================================================
    def check_first_crossing(
        self,
        guidance_t,
        mission_t,
    ):
        d = self.gate_signed_distance(
            self.current_x,
            self.current_y,
        )

        crossed = (
            self.prev_gate_d is not None
            and self.prev_gate_d < 0.0
            and d >= 0.0
        )
        # Gate completion definition: only the first physical crossing
        # of the gate line is accepted. Range R is never used here.

        if not crossed:
            self.prev_gate_d = d
            self.prev_guidance_t = guidance_t
            self.prev_mission_t = mission_t
            self.prev_x = self.current_x
            self.prev_y = self.current_y
            self.prev_yaw = self.current_yaw
            return False

        den = d - self.prev_gate_d
        if abs(den) < 1.0e-12:
            alpha = 0.0
        else:
            alpha = -self.prev_gate_d / den

        alpha = self.sat(alpha, 0.0, 1.0)

        t_cross_mission = (
            self.prev_mission_t
            + alpha * (mission_t - self.prev_mission_t)
        )

        x_cross = (
            self.prev_x
            + alpha * (self.current_x - self.prev_x)
        )
        y_cross = (
            self.prev_y
            + alpha * (self.current_y - self.prev_y)
        )

        dyaw = self.normalize_angle(
            self.current_yaw - self.prev_yaw
        )
        yaw_cross = self.normalize_angle(
            self.prev_yaw + alpha * dyaw
        )

        # Gate tangent direction.
        tx = -math.sin(self.lambda_desired)
        ty = math.cos(self.lambda_desired)

        e_p = (
            tx * (x_cross - self.target_x)
            + ty * (y_cross - self.target_y)
        )
        e_theta = self.normalize_angle(
            yaw_cross - self.lambda_desired
        )
        e_T = t_cross_mission - self.desired_total_time

        self.crossing_finished = True
        self.cross_time = t_cross_mission
        self.cross_eT = e_T
        self.cross_ep = e_p
        self.cross_etheta = e_theta

        rospy.loginfo("========== FIRST GATE CROSSING ==========")
        rospy.loginfo(
            "t_cross = %.6f s | e_T = %+.6f s",
            t_cross_mission,
            e_T,
        )
        rospy.loginfo(
            "e_p = %+.6f m | e_theta = %+.6f deg",
            e_p,
            math.degrees(e_theta),
        )
        rospy.loginfo(
            "v_cmd = %.6f m/s",
            self.cruise_speed,
        )
        rospy.loginfo("=========================================")

        return True

    # ================================================================
    # Li-style start-line crossing: task clock begins at y=1.5 crossing
    # ================================================================
    def check_start_crossing(self, now_sec):
        if self.prev_start_y is None:
            self.prev_start_y = self.current_y
            self.prev_start_x = self.current_x
            self.prev_start_time = now_sec
            return False

        crossed = self.prev_start_y > self.start_y and self.current_y <= self.start_y
        if not crossed:
            self.prev_start_y = self.current_y
            self.prev_start_x = self.current_x
            self.prev_start_time = now_sec
            return False

        den = self.current_y - self.prev_start_y
        alpha = 0.0 if abs(den) < 1e-9 else (self.start_y - self.prev_start_y) / den
        alpha = self.sat(alpha, 0.0, 1.0)
        x_cross = self.prev_start_x + alpha * (self.current_x - self.prev_start_x)
        rospy.loginfo("========== WANG TASK START ==========")
        rospy.loginfo("start crossing: x=%.4f y=%.4f", x_cross, self.start_y)
        rospy.loginfo("20 s task clock starts now")
        rospy.loginfo("=====================================")
        return True

    def print_final_metrics(self):
        rospy.loginfo("============== WANG FINAL METRICS ==============")
        if self.crossing_finished:
            rospy.loginfo("t_cross       = %.6f s", self.cross_time)
            rospy.loginfo("e_T           = %+ .6f s", self.cross_eT)
            rospy.loginfo("e_p           = %+ .6f m", self.cross_ep)
            rospy.loginfo("e_theta       = %+ .6f deg",
                          math.degrees(self.cross_etheta))
        else:
            rospy.logwarn("No valid gate crossing detected.")
        R, los, sigma, lambda_e, varpi = self.get_geometry()
        rospy.loginfo("final position = (%.6f, %.6f)", self.current_x, self.current_y)
        rospy.loginfo("final R        = %.6f m", R)
        rospy.loginfo("final yaw      = %.6f deg", math.degrees(self.current_yaw))
        rospy.loginfo("final sigma    = %.6f deg", math.degrees(sigma))
        rospy.loginfo("final lambda_e = %.6f deg", math.degrees(lambda_e))
        rospy.loginfo("final varpi    = %.6f deg", math.degrees(varpi))
        rospy.loginfo("===============================================")

    # ================================================================
    # Main loop
    # ================================================================
    def run(self):
        rospy.loginfo("Waiting for motion-capture pose...")

        try:
            msg = rospy.wait_for_message(
                "/vrpn_client_node/car1/pose",
                PoseStamped,
                timeout=5.0,
            )
            self.pose_callback(msg)
        except rospy.ROSException:
            rospy.logerr("No motion-capture pose. Node exits.")
            return

        self.node_start = rospy.Time.now()
        self.last_time_sec = self.node_start.to_sec()

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            now_sec = now.to_sec()

            dt_real = now_sec - self.last_time_sec
            if dt_real <= 0.001 or dt_real > 0.2:
                dt_real = 0.02
            self.last_time_sec = now_sec

            mission_t = 0.0 if self.mission_start is None else (now - self.mission_start).to_sec()

            # --------------------------------------------------------
            # State 1: Li-style run-in. Place vehicle at (2,2), move down,
            # and start the 20 s Wang task clock only after y=1.5 crossing.
            # --------------------------------------------------------
            if self.state == self.STATE_RUNIN:
                V_cmd = self.cruise_speed * min(1.0, (now - self.node_start).to_sec() / self.accel_duration)
                omega_cmd = self.publish_cmd(V_cmd, 0.0, dt_real)
                if self.check_start_crossing(now_sec):
                    try:
                        self.initialize_wang()
                    except RuntimeError as exc:
                        rospy.logerr(str(exc))
                        self.publish_cmd_direct(0.0, 0.0)
                        return
                    self.mission_start = now
                    self.guidance_start = now
                    self.state = self.STATE_GUIDANCE
                telemetry = {
                    "mission_time": mission_t,
                    "guidance_time": 0.0,
                    "speed_actual": self.actual_speed,
                    "speed_desired": V_cmd,
                    "omega_raw": 0.0,
                    "omega_cmd": omega_cmd,
                    "yaw_actual": self.current_yaw,
                    "pos_x": self.current_x,
                    "pos_y": self.current_y,
                }
                self.record_csv(telemetry)
                self.rate.sleep()
                continue

            # --------------------------------------------------------
            # State 2: Wang constant-speed ITACG
            # --------------------------------------------------------
            if self.state == self.STATE_GUIDANCE:
                guidance_t = (
                    now - self.guidance_start
                ).to_sec()

                # Same optional disturbance convention used by the
                # current TIE experiment code.
                disturbance = 0.0
                if (
                    self.inject_disturbance
                    and self.dist_start_time
                    <= guidance_t
                    <= self.dist_start_time
                    + self.dist_duration
                ):
                    disturbance = self.dist_omega_val
                    rospy.logwarn_throttle(
                        0.2,
                        "Wang disturbance test: t=%.2f s, "
                        "d_omega=%.2f rad/s",
                        guidance_t,
                        disturbance,
                    )

                try:
                    omega_raw, data = self.calc_wang_command(
                        guidance_t
                    )
                except RuntimeError as exc:
                    rospy.logerr(
                        "Wang controller safety stop: %s",
                        str(exc),
                    )
                    self.publish_cmd_direct(0.0, 0.0)
                    return

                omega_to_robot = omega_raw + disturbance

                omega_cmd = self.publish_cmd(
                    self.cruise_speed,
                    omega_to_robot,
                    dt_real,
                )

                telemetry = {
                    "mission_time": mission_t,
                    "guidance_time": guidance_t,
                    "relative_dist": data["R"],
                    "tgo_desired": data["tgo_desired"],
                    "tgo_model": data["tgo_model"],
                    "e_tgo": data["e_tgo"],
                    "speed_actual": self.actual_speed,
                    "speed_desired": self.cruise_speed,
                    "sigma_actual": data["sigma"],
                    "lambda_actual": data["los"],
                    "lambda_error": data["lambda_e"],
                    "varpi": data["varpi"],
                    "K_e": data["K_e"],
                    "A_iac": data["A_iac"],
                    "A_bias": data["A_bias"],
                    "A_itac": data["A_itac"],
                    "omega_raw": omega_raw,
                    "omega_cmd": omega_cmd,
                    "yaw_actual": self.current_yaw,
                    "pos_x": self.current_x,
                    "pos_y": self.current_y,
                }

                self.publish_telemetry(telemetry)
                self.record_csv(telemetry)

                if self.check_terminal_lock(data):
                    self.state = self.STATE_LOCK
                    rospy.loginfo("Late terminal lock activated. Continue until dg=0 crossing.")

                if self.check_first_crossing(
                    guidance_t,
                    mission_t,
                ):
                    self.state = self.STATE_TERMINAL
                    self.publish_cmd_direct(0.0, 0.0)
                    rospy.loginfo("Wang gate-passing task finished: first dg crossing detected.")
                    break

                # Safety timeout.
                if mission_t > self.desired_total_time + 5.0:
                    rospy.logerr(
                        "No gate crossing before timeout."
                    )
                    self.publish_cmd_direct(0.0, 0.0)
                    break

            # --------------------------------------------------------
            # State 3: terminal straight-line heading lock
            # --------------------------------------------------------
            elif self.state == self.STATE_LOCK:
                omega_lock = self.calc_terminal_lock_command()
                omega_cmd = self.publish_cmd(
                    self.cruise_speed,
                    omega_lock,
                    dt_real
                )

                R, los, sigma, lambda_e, varpi = self.get_geometry()
                telemetry = {
                    "mission_time": mission_t,
                    "guidance_time": mission_t - self.accel_duration,
                    "relative_dist": R,
                    "sigma_actual": sigma,
                    "lambda_actual": los,
                    "lambda_error": lambda_e,
                    "varpi": varpi,
                    "omega_raw": omega_lock,
                    "omega_cmd": omega_cmd,
                    "yaw_actual": self.current_yaw,
                    "pos_x": self.current_x,
                    "pos_y": self.current_y,
                }
                self.record_csv(telemetry)

                if self.check_first_crossing(
                    mission_t - self.accel_duration,
                    mission_t,
                ):
                    self.state = self.STATE_TERMINAL
                    self.publish_cmd_direct(0.0, 0.0)
                    self.print_final_metrics()
                    break

            # --------------------------------------------------------
            # State 4: terminal stop AFTER FIRST PHYSICAL GATE CROSSING ONLY
            # --------------------------------------------------------
            elif self.state == self.STATE_TERMINAL:
                self.publish_cmd_direct(0.0, 0.0)
                break

            self.rate.sleep()


if __name__ == "__main__":
    try:
        WangITACGGuidance().run()
    except rospy.ROSInterruptException:
        pass