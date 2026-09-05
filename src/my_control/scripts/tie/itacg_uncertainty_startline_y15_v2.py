#!/usr/bin/env python3
# coding: utf-8 

import rospy
import math
import tf
import csv
import os
from datetime import datetime

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry  
from std_msgs.msg import Float64

class ImpactTimeControlGuidance:
    def __init__(self):
        rospy.init_node('impact_time_control_guidance', anonymous=True)
        
        # 安全场景：目标点左移，缩减初始相对距离以控制最大外扩曲率
        self.target_x = -1.5
        self.target_y = 1.5
        
        # 1. 平台运动学约束与极限参数 
        self.cruise_speed = 0.4
        self.terminal_speed = 0.2  
        
        self.v_min = 0.05    
        self.v_max = 0.50    
        self.a_f = 0.5       
        self.a_b = 0.5       
        self.Delta_L_max = 5.0 
        self.chi_min = 0.10                 # 初始剩余路程裕度
        self.eta_min = 0.10                 # S0 插值下限权重
        self.eta_max = 0.6                # S0 插值上限权重
        self.sigma_sat = math.radians(120.0) # 初始前置角饱和值
        
        # self.distance_budget_factor = 1.65  # 旧版比例预算参数，新版 S0 选择不再使用
        self.max_angular = 3.0
        self.max_yaw_accel = 2.0     
        # Run-in acceleration is outside the task clock.  The vehicle is
        # placed around y=2.0, accelerates rapidly, and the 20-s mission
        # clock starts only at the downward crossing of y=1.5.
        self.runin_accel_duration = 0.20
        self.start_line_y = 1.5
        self.runin_target_yaw = math.radians(-90.0)
        self.runin_heading_gain = 1.5
        self.start_speed_warn = 0.03
        self.distance_threshold = 0.05
        
        # 2. 制导律参数与平滑滤波设置
        # IMPORTANT: acceleration is now completed before the start line,
        # therefore the full prescribed horizon is 20.0 s after y=1.5.
        self.desired_impact_time = 20.0
        self.lambda_desired = math.radians(135.0)
        
        self.K_f = 2.1              
        self.n = 1.5                 
        self.tau = 0.16
        self.a_gain = 1.0    
        self.nav_gain = 3.0  
        
        self.lambda_corr_gain = 0.35           
        self.sigma_lambda_max = math.radians(10.0)

        # Angular-error recovery safeguard: only blends in when chi is nearly exhausted
        # while the gate-heading error is still non-negligible. No speed compensation is used.
        self.chi_recovery_th = 0.05
        self.lambda_recovery_th = math.radians(2.0)
        self.lambda_recovery_band = math.radians(8.0)
        
        # 终段触发阈值：新增 2度 角度锁死阈值
        self.terminal_angle_error = math.radians(3.0)
        self.terminal_blind_tgo = 1.0   
        self.terminal_blind_R = 0.4     
        self.locked_yaw = 0.0  

        self.sigma_dot_limit = math.radians(35.0)
        self.R_los_min = 0.18
        self.omega_slew_limit = 1.5     
        self.last_omega_cmd = 0.0
        self.last_cmd_omega = 0.0       
        self.omega_filter_alpha = 0.1 
        
        # [核心新增] 前置角微分专用的平滑滤波器，彻底消除指令高频毛刺
        self.sigma_dot_filter_alpha = 0.2 
        self.filtered_sigma_dot = 0.0
        
        self.K_s = 2.5 
        self.t_s = 0.0 
        self.filtered_sigma_d = 0.0
        self.filtered_omega = 0.0       
        self.current_x = self.current_y = self.current_yaw = self.last_los = self.actual_speed = 0.0
        self.actual_yaw_rate = 0.0

        self.STATE_RUNIN, self.STATE_ITACG, self.STATE_PNG, self.STATE_TERMINAL = 0, 2, 3, 4
        self.state = self.STATE_RUNIN
        self.last_time_sec = None
        
        # ================= 测试用：角速度执行增益失配 =================
        # Controller calculates omega_c internally, while the robot receives:
        # omega_app = alpha_omega * omega_c
        # This models steering effectiveness / low-level angular-rate gain error.
        # Nominal: alpha_omega = 1.0
        # R2:      alpha_omega = 0.85
        # R3:      alpha_omega = 1.15
        self.alpha_omega = float(rospy.get_param('~alpha_omega', 1.0))

        # Optional additive external disturbance.  It is disabled by default,
        # but the fields are always logged so the same CSV format can be used
        # for nominal, model-uncertainty, and disturbance trials.
        self.inject_disturbance = bool(rospy.get_param('~inject_disturbance', False))
        self.dist_start_time = float(rospy.get_param('~dist_start_time', 5.0))
        self.dist_duration = float(rospy.get_param('~dist_duration', 1.0))
        self.dist_omega_val = float(rospy.get_param('~dist_omega_val', 0.5))
        # ==================================================
        
        # ================= CSV 记录初始化 =================
        log_dir = os.path.expanduser('~')
        filename = datetime.now().strftime("itacg_data_%Y%m%d_%H%M%S.csv")
        self.csv_path = os.path.join(log_dir, filename)
        self.csv_file = open(self.csv_path, 'w')
        self.csv_writer = csv.writer(self.csv_file)
        
        # 定义要按序保存的字段名
        self.csv_keys = [
            'node_time', 'task_started', 'state',
            'relative_dist', 'tgo', 'speed_actual', 'speed_desired',
            'sigma_actual', 'sigma_desired', 'lambda_actual', 'lambda_error',
            'yaw_actual', 'angular_vel', 'yaw_rate_actual',
            'sigma_lambda', 'sigma_dot', 'los_rate_raw',
            'omega_raw', 'omega_nominal', 'alpha_omega',
            'omega_uncertain', 'dist_omega', 'omega_requested',
            'omega_applied', 'omega_cmd',
            'pos_x', 'pos_y'
        ]
        headers = ['time'] + [k + '_deg' if any(x in k for x in ['sigma', 'lambda', 'yaw']) else k for k in self.csv_keys]
        self.csv_writer.writerow(headers)
        rospy.on_shutdown(self.shutdown_hook)
        # ==================================================
        
        rospy.Subscriber('/vrpn_client_node/car1/pose', PoseStamped, self.pose_callback)
        rospy.Subscriber('/car1/odom', Odometry, self.odom_callback) 
        self.cmd_pub = rospy.Publisher('/car1/cmd_vel', Twist, queue_size=10)
        
        topics = ['angular_vel', 'yaw_rate_actual', 'sigma_actual', 'sigma_desired',
                  'lambda_actual', 'lambda_error', 'relative_dist', 'tgo',
                  'speed_desired', 'speed_actual', 'sigma_lambda', 'sigma_dot',
                  'los_rate_raw', 'omega_raw', 'omega_nominal', 'alpha_omega',
                  'omega_uncertain', 'dist_omega', 'omega_requested',
                  'omega_applied', 'omega_cmd', 'yaw_actual']
                  
        self.pubs = {name: rospy.Publisher('/itacg/{}'.format(name), Float64, queue_size=10) for name in topics}
        self.rate = rospy.Rate(50)
        rospy.loginfo(
            "SBHAG节点初始化完毕 | alpha_omega=%.3f | start line y=%.2f | T=%.2fs | CSV=%s",
            self.alpha_omega, self.start_line_y, self.desired_impact_time, self.csv_path)

    def shutdown_hook(self):
        if not self.csv_file.closed:
            self.csv_file.close()
            rospy.loginfo("节点已关闭，CSV数据文件已成功保存至: %s", self.csv_path)

    def record_csv(self, t, telemetry):
        row = [t]
        for key in self.csv_keys:
            val = telemetry.get(key, 0.0)  
            if any(x in key for x in ['sigma', 'lambda', 'yaw']):
                val = math.degrees(val)
            row.append(val)
        self.csv_writer.writerow(row)

    def sat(self, x, min_val, max_val):
        return max(min_val, min(max_val, x))

    def slew_limit(self, x, x_last, rate, dt):
        return x_last + self.sat(x - x_last, -rate * dt, rate * dt)

    def normalize_angle(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi
    
    def get_line_of_sight_angle(self):
        return math.atan2(self.target_y - self.current_y, self.target_x - self.current_x)

    def calc_S_max(self, v0, vg, td):
        t_acc = (self.v_max - v0) / self.a_f
        t_dec = (self.v_max - vg) / self.a_b
        if t_acc + t_dec <= td:
            t_cr = td - t_acc - t_dec
            return (v0 + self.v_max)/2 * t_acc + self.v_max * t_cr + (self.v_max + vg)/2 * t_dec
        t1 = (vg - v0 + self.a_b * td) / (self.a_f + self.a_b)
        vp = v0 + self.a_f * t1
        return (v0 + vp)/2 * t1 + (vp + vg)/2 * (td - t1)

    def calc_S_min(self, v0, vg, td):
        t_dec = (v0 - self.v_min) / self.a_b
        t_acc = (vg - self.v_min) / self.a_f
        if t_dec + t_acc <= td:
            t_cr = td - t_dec - t_acc
            return (v0 + self.v_min)/2 * t_dec + self.v_min * t_cr + (self.v_min + vg)/2 * t_acc
        t1 = (v0 - vg + self.a_f * td) / (self.a_b + self.a_f)
        vv = v0 - self.a_b * t1
        return (v0 + vv)/2 * t1 + (vv + vg)/2 * (td - t1)

    def mod2pi(self, x):
        return x - 2.0 * math.pi * math.floor(x / (2.0 * math.pi))

    def dubins_length(self, x0, y0, yaw0, x1, y1, yaw1, rho):
        if rho <= 1e-6:
            return math.hypot(x1 - x0, y1 - y0)

        dx, dy = x1 - x0, y1 - y0
        d = math.hypot(dx, dy) / rho
        if d < 1e-9:
            return abs(self.normalize_angle(yaw1 - yaw0)) * rho

        theta = self.mod2pi(math.atan2(dy, dx))
        alpha = self.mod2pi(yaw0 - theta)
        beta = self.mod2pi(yaw1 - theta)
        sa, sb = math.sin(alpha), math.sin(beta)
        ca, cb = math.cos(alpha), math.cos(beta)
        cab = math.cos(alpha - beta)
        cand = []

        p2 = 2 + d*d - 2*cab + 2*d*(sa - sb)
        if p2 >= 0:
            tmp = math.atan2(cb - ca, d + sa - sb)
            cand.append(self.mod2pi(-alpha + tmp) + math.sqrt(p2) + self.mod2pi(beta - tmp))

        p2 = 2 + d*d - 2*cab + 2*d*(-sa + sb)
        if p2 >= 0:
            tmp = math.atan2(ca - cb, d - sa + sb)
            cand.append(self.mod2pi(alpha - tmp) + math.sqrt(p2) + self.mod2pi(-beta + tmp))

        p2 = -2 + d*d + 2*cab + 2*d*(sa + sb)
        if p2 >= 0:
            p = math.sqrt(p2)
            tmp = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p)
            cand.append(self.mod2pi(-alpha + tmp) + p + self.mod2pi(-beta + tmp))

        p2 = -2 + d*d + 2*cab - 2*d*(sa + sb)
        if p2 >= 0:
            p = math.sqrt(p2)
            tmp = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, p)
            cand.append(self.mod2pi(alpha - tmp) + p + self.mod2pi(beta - tmp))

        tmp = (6 - d*d + 2*cab + 2*d*(sa - sb)) / 8.0
        if abs(tmp) <= 1:
            p = self.mod2pi(2.0 * math.pi - math.acos(tmp))
            t = self.mod2pi(alpha - math.atan2(ca - cb, d - sa + sb) + 0.5 * p)
            cand.append(t + p + self.mod2pi(alpha - beta - t + p))

        tmp = (6 - d*d + 2*cab + 2*d*(-sa + sb)) / 8.0
        if abs(tmp) <= 1:
            p = self.mod2pi(2.0 * math.pi - math.acos(tmp))
            t = self.mod2pi(-alpha - math.atan2(ca - cb, d + sa - sb) + 0.5 * p)
            cand.append(t + p + self.mod2pi(beta - alpha - t + p))

        return (min(cand) * rho) if cand else math.hypot(dx, dy)
    
    def publish_telemetry(self, data_dict):
        for key, val in data_dict.items():
            if key in self.pubs:
                if any(x in key for x in ['sigma', 'lambda', 'yaw']):
                    self.pubs[key].publish(math.degrees(val))
                else:
                    self.pubs[key].publish(val)

    def odom_callback(self, msg):
        self.actual_speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        self.actual_yaw_rate = msg.twist.twist.angular.z
        
    def pose_callback(self, msg):
        self.current_x, self.current_y = msg.pose.position.x, msg.pose.position.y
        q = msg.pose.orientation
        _, _, self.current_yaw = tf.transformations.euler_from_quaternion((q.x, q.y, q.z, q.w))
    
    def publish_cmd(self, v, omega, dt):
        omega = self.slew_limit(omega, self.last_cmd_omega, self.max_yaw_accel, dt)
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = self.sat(omega, -self.max_angular, self.max_angular)
        self.last_cmd_omega = cmd.angular.z
        self.cmd_pub.publish(cmd)
        return cmd.angular.z

    def interpolate_angle(self, a0, a1, r):
        return self.normalize_angle(a0 + r * self.normalize_angle(a1 - a0))

    def initialize_guidance_at_start(self, x0, y0, yaw0, v0_measured):
        """Initialize S0, Ks, filters, etc. at the actual y=1.5 crossing."""
        R_0 = math.hypot(self.target_x - x0, self.target_y - y0)
        los_0 = math.atan2(self.target_y - y0, self.target_x - x0)
        sigma_0 = self.normalize_angle(yaw0 - los_0)
        lambda_e_0 = self.normalize_angle(los_0 - self.lambda_desired)

        self.T_total = self.desired_impact_time
        self.v_0 = self.cruise_speed
        self.v_g = self.terminal_speed

        S_min = self.calc_S_min(self.v_0, self.v_g, self.T_total)
        S_max = self.calc_S_max(self.v_0, self.v_g, self.T_total)
        rho = self.v_max / max(self.max_angular, 1e-6)
        L_min = self.dubins_length(
            x0, y0, yaw0,
            self.target_x, self.target_y, self.lambda_desired, rho)
        L_max = L_min + self.Delta_L_max

        S_lower = max(S_min, L_min, R_0 + self.chi_min)
        S_upper = min(S_max, L_max)

        sigma_bar = min(abs(self.normalize_angle(sigma_0)), self.sigma_sat)
        eta_den = max(1e-6, 1.0 - math.cos(self.sigma_sat))
        eta_S = self.eta_min + (self.eta_max - self.eta_min) * \
            (1.0 - math.cos(sigma_bar)) / eta_den
        eta_S = self.sat(eta_S, 0.0, 1.0)

        rospy.loginfo(
            "start-line physical interval: [%.3f, %.3f] m | geometry: [%.3f, %.3f] m",
            S_min, S_max, L_min, L_max)

        if S_lower > S_upper:
            rospy.logwarn("No compatible distance interval at the start line; using midpoint fallback.")
            self.S_0 = 0.5 * (S_min + S_max)
        else:
            self.S_0 = (1.0 - eta_S) * S_lower + eta_S * S_upper

        self.c_coeff = 30.0 * (
            self.S_0 / self.T_total - (self.v_0 + self.v_g) / 2.0)

        chi_0 = self.S_0 - R_0
        if abs(chi_0) > 1e-4:
            self.K_s = abs(
                (R_0 * lambda_e_0 / chi_0) * math.tan(sigma_0 / 2.0))
        else:
            self.K_s = self.K_f

        self.filtered_sigma_d = sigma_0
        self.filtered_sigma_dot = 0.0
        self.filtered_omega = 0.0
        self.last_los = los_0
        self.state = self.STATE_ITACG

        rospy.loginfo("========== ITACG TASK INITIALIZATION ==========")
        rospy.loginfo(
            "start: x=%.4f y=%.4f yaw=%.3f deg v=%.4f m/s",
            x0, y0, math.degrees(yaw0), v0_measured)
        rospy.loginfo(
            "R0=%.4f m sigma0=%.3f deg lambda_e0=%.3f deg",
            R_0, math.degrees(sigma_0), math.degrees(lambda_e_0))
        rospy.loginfo(
            "S interval=[%.4f, %.4f] m eta=%.3f S0=%.4f m Ks=%.4f",
            S_lower, S_upper, eta_S, self.S_0, self.K_s)
        if abs(v0_measured - self.v_0) > self.start_speed_warn:
            rospy.logwarn(
                "start-line speed differs from v0 by %.4f m/s (measured %.4f, desired %.4f)",
                v0_measured - self.v_0, v0_measured, self.v_0)
        rospy.loginfo("================================================")

    def disturbance_value(self, t):
        if (self.inject_disturbance and
                self.dist_start_time <= t < self.dist_start_time + self.dist_duration):
            return self.dist_omega_val
        return 0.0

    def run(self):
        rospy.loginfo("等待动捕数据...")
        try:
            self.pose_callback(
                rospy.wait_for_message(
                    '/vrpn_client_node/car1/pose', PoseStamped, timeout=5.0))
        except rospy.ROSException:
            return rospy.logerr("未收到动捕数据，节点退出")

        rospy.loginfo("等待里程计数据...")
        try:
            self.odom_callback(
                rospy.wait_for_message(
                    '/car1/odom', Odometry, timeout=5.0))
        except rospy.ROSException:
            return rospy.logerr("未收到里程计数据，节点退出")

        if self.current_y <= self.start_line_y + 0.03:
            rospy.logerr(
                "初始位置必须在起跑线 y=%.2f 上方。建议摆放在 x≈2.0, y≈2.0, yaw≈-90deg。当前 y=%.3f",
                self.start_line_y, self.current_y)
            return

        node_start_sec = rospy.Time.now().to_sec()
        self.last_time_sec = node_start_sec

        rospy.loginfo("========== RUN-IN ==========")
        rospy.loginfo(
            "建议摆放: x≈2.0 m, y≈2.0 m, yaw≈-90 deg | 当前 x=%.3f y=%.3f yaw=%.2f deg",
            self.current_x, self.current_y, math.degrees(self.current_yaw))
        rospy.loginfo(
            "run-in 加速时间 %.2f s；y=%.2f 之前不计入任务时间；过线后直接以 v0=%.2f m/s 进入 ITACG",
            self.runin_accel_duration, self.start_line_y, self.cruise_speed)
        rospy.loginfo("============================")

        # Previous sample for exact line-crossing interpolation.
        prev_sec = node_start_sec
        prev_x = self.current_x
        prev_y = self.current_y
        prev_yaw = self.current_yaw
        prev_speed = self.actual_speed

        task_start_sec = None

        # ------------------------------------------------------------
        # RUN-IN: fast acceleration and heading hold; task clock OFF.
        # We intentionally do not write these rows to the task CSV so
        # time=0 in the CSV corresponds to the y=1.5 crossing.
        # ------------------------------------------------------------
        while not rospy.is_shutdown() and task_start_sec is None:
            now_sec = rospy.Time.now().to_sec()
            dt_real = now_sec - self.last_time_sec
            if dt_real <= 0.001 or dt_real > 0.2:
                dt_real = 0.02
            self.last_time_sec = now_sec

            runin_t = now_sec - node_start_sec
            ramp = self.sat(
                runin_t / max(self.runin_accel_duration, 1e-6), 0.0, 1.0)
            V_cmd = self.cruise_speed * ramp

            # Run-in is only an initialization maneuver; uncertainty is
            # activated from task t=0 so all robustness trials share the
            # same intended initial condition.
            yaw_err = self.normalize_angle(self.runin_target_yaw - self.current_yaw)
            omega_runin = self.runin_heading_gain * yaw_err
            self.publish_cmd(V_cmd, omega_runin, dt_real)

            crossed = prev_y > self.start_line_y and self.current_y <= self.start_line_y
            if crossed:
                den = self.current_y - prev_y
                if abs(den) < 1e-9:
                    frac = 0.0
                else:
                    frac = (self.start_line_y - prev_y) / den
                frac = self.sat(frac, 0.0, 1.0)

                cross_sec = prev_sec + frac * (now_sec - prev_sec)
                x_start = prev_x + frac * (self.current_x - prev_x)
                yaw_start = self.interpolate_angle(prev_yaw, self.current_yaw, frac)
                v_start = prev_speed + frac * (self.actual_speed - prev_speed)

                task_start_sec = cross_sec
                self.initialize_guidance_at_start(
                    x_start, self.start_line_y, yaw_start, v_start)

                rospy.loginfo("========== ITACG TASK START ==========")
                rospy.loginfo(
                    "interpolated crossing: x=%.4f y=%.4f yaw=%.3f deg speed=%.4f m/s",
                    x_start, self.start_line_y, math.degrees(yaw_start), v_start)
                rospy.loginfo(
                    "20.000-s clock starts at y=%.2f. alpha_omega=%.3f",
                    self.start_line_y, self.alpha_omega)
                rospy.loginfo("======================================")
                break

            prev_sec = now_sec
            prev_x = self.current_x
            prev_y = self.current_y
            prev_yaw = self.current_yaw
            prev_speed = self.actual_speed
            self.rate.sleep()

        if rospy.is_shutdown() or task_start_sec is None:
            return

        # ------------------------------------------------------------
        # TASK: y=1.5 crossing is t=0.  No post-start acceleration reset.
        # ------------------------------------------------------------
        while not rospy.is_shutdown():
            now_sec = rospy.Time.now().to_sec()
            t = now_sec - task_start_sec
            node_t = now_sec - node_start_sec

            dt_real = now_sec - self.last_time_sec
            if dt_real <= 0.001 or dt_real > 0.2:
                dt_real = 0.02
            self.last_time_sec = now_sec

            t_go = self.T_total - t
            alpha_omega = self.alpha_omega

            R = math.hypot(
                self.target_x - self.current_x,
                self.target_y - self.current_y)
            los = self.get_line_of_sight_angle()
            sigma = self.normalize_angle(self.current_yaw - los)
            lambda_e = self.normalize_angle(los - self.lambda_desired)

            telemetry = {
                'node_time': node_t,
                'task_started': 1.0,
                'state': float(self.state),
                'relative_dist': R,
                'tgo': t_go,
                'speed_actual': self.actual_speed,
                'sigma_actual': sigma,
                'lambda_actual': los,
                'lambda_error': lambda_e,
                'yaw_actual': self.current_yaw,
                'yaw_rate_actual': self.actual_yaw_rate,
                'pos_x': self.current_x,
                'pos_y': self.current_y,
                'alpha_omega': alpha_omega,
                'omega_uncertain': 0.0,
                'dist_omega': 0.0,
                'omega_requested': 0.0,
                'omega_applied': self.last_cmd_omega,
                'omega_cmd': self.last_cmd_omega,
            }

            if self.state == self.STATE_ITACG:
                if ((R < 0.6 and abs(lambda_e) <= self.terminal_angle_error)
                        or t_go <= self.terminal_blind_tgo
                        or R <= self.terminal_blind_R):
                    self.state = self.STATE_PNG
                    self.locked_yaw = self.lambda_desired
                    self.filtered_omega = 0.0
                    rospy.loginfo(
                        "满足终段触发条件，切换至目标偏航角锁死模式。")
                    continue

                s = self.sat(t / self.T_total, 0.0, 1.0)
                V_cmd = (
                    self.v_0
                    + (self.v_g - self.v_0) * (3*s**2 - 2*s**3)
                    + self.c_coeff * (s**2 * (1-s)**2))

                F_s = (
                    self.v_0*s
                    + (self.v_g - self.v_0) * (s**3 - 0.5*s**4)
                    + self.c_coeff * (s**3/3.0 - 0.5*s**4 + s**5/5.0))
                S_r = self.T_total * (
                    0.5 * (self.v_0 + self.v_g)
                    + self.c_coeff/30.0
                    - F_s)
                chi_e = max(S_r - R, 0.0)

                if t_go > 0.0:
                    K_t = self.K_f + (self.K_s - self.K_f) * (
                        max(0.0, t_go / self.T_total) ** self.n)
                else:
                    K_t = self.K_f

                lambda_safe = (
                    lambda_e if abs(lambda_e) > 0.01
                    else math.copysign(0.01, lambda_e if lambda_e != 0 else 1.0))
                denom = max(R, 0.05) * lambda_safe
                sigma_d_0 = 2.0 * math.atan((K_t * chi_e) / denom)

                V_eff = max(V_cmd, 0.08)
                R_corr = max(R, 0.15)
                fade_distance = 0.8
                if R < fade_distance:
                    current_sigma_max = self.sigma_lambda_max * (R / fade_distance)
                else:
                    current_sigma_max = self.sigma_lambda_max

                sin_sigma_lam = self.sat(
                    self.lambda_corr_gain * R_corr * lambda_e / V_eff,
                    -math.sin(current_sigma_max),
                    math.sin(current_sigma_max))
                sigma_lam = math.asin(sin_sigma_lam)

                sigma_HA = sigma_d_0
                mu_chi = self.sat(
                    (self.chi_recovery_th - chi_e) / max(self.chi_recovery_th, 1e-6),
                    0.0, 1.0)
                mu_lam = self.sat(
                    (abs(lambda_e) - self.lambda_recovery_th) /
                    max(self.lambda_recovery_band, 1e-6),
                    0.0, 1.0)
                w_chi = mu_chi * mu_chi * (3.0 - 2.0 * mu_chi)
                w_lam = mu_lam * mu_lam * (3.0 - 2.0 * mu_lam)
                w_rec = w_chi * w_lam
                sigma_d_0 = self.normalize_angle(
                    sigma_HA + w_rec * self.normalize_angle(sigma_lam - sigma_HA))

                raw_sigma_dot = self.sat(
                    self.normalize_angle(sigma_d_0 - self.filtered_sigma_d) / self.tau,
                    -self.sigma_dot_limit,
                    self.sigma_dot_limit)
                self.filtered_sigma_dot = (
                    self.sigma_dot_filter_alpha * raw_sigma_dot
                    + (1.0 - self.sigma_dot_filter_alpha) * self.filtered_sigma_dot)
                self.filtered_sigma_d = self.normalize_angle(
                    self.filtered_sigma_d + self.filtered_sigma_dot * dt_real)

                los_rate_raw = (
                    -V_cmd * math.sin(sigma) / max(R, self.R_los_min))
                raw_omega = (
                    los_rate_raw
                    + self.filtered_sigma_dot
                    - self.a_gain * self.normalize_angle(sigma - self.filtered_sigma_d))
                self.filtered_omega = (
                    self.omega_filter_alpha * raw_omega
                    + (1.0 - self.omega_filter_alpha) * self.filtered_omega)

                omega_nominal = self.filtered_omega
                omega_uncertain = alpha_omega * omega_nominal
                dist_omega = self.disturbance_value(t)
                omega_requested = omega_uncertain + dist_omega
                omega_applied = self.publish_cmd(V_cmd, omega_requested, dt_real)

                telemetry.update({
                    'speed_desired': V_cmd,
                    'sigma_desired': self.filtered_sigma_d,
                    # Keep angular_vel as applied command for compatibility
                    # with the previous CSV format.
                    'angular_vel': omega_applied,
                    'sigma_lambda': sigma_lam,
                    'sigma_dot': self.filtered_sigma_dot,
                    'los_rate_raw': los_rate_raw,
                    'omega_raw': raw_omega,
                    'omega_nominal': omega_nominal,
                    'omega_uncertain': omega_uncertain,
                    'dist_omega': dist_omega,
                    'omega_requested': omega_requested,
                    'omega_applied': omega_applied,
                    'omega_cmd': omega_applied,
                    'state': float(self.state),
                })

            elif self.state == self.STATE_PNG:
                s = self.sat(t / self.T_total, 0.0, 1.0)
                V_cmd = (
                    self.v_0
                    + (self.v_g - self.v_0) * (3*s**2 - 2*s**3)
                    + self.c_coeff * (s**2 * (1-s)**2))

                yaw_error = self.normalize_angle(self.locked_yaw - self.current_yaw)
                raw_omega = 0.5 * yaw_error
                self.filtered_omega = (
                    self.omega_filter_alpha * raw_omega
                    + (1.0 - self.omega_filter_alpha) * self.filtered_omega)

                omega_nominal = self.filtered_omega
                omega_uncertain = alpha_omega * omega_nominal
                dist_omega = self.disturbance_value(t)
                omega_requested = omega_uncertain + dist_omega
                omega_applied = self.publish_cmd(V_cmd, omega_requested, dt_real)

                telemetry.update({
                    'speed_desired': V_cmd,
                    'sigma_desired': 0.0,
                    'angular_vel': omega_applied,
                    'omega_raw': raw_omega,
                    'omega_nominal': omega_nominal,
                    'omega_uncertain': omega_uncertain,
                    'dist_omega': dist_omega,
                    'omega_requested': omega_requested,
                    'omega_applied': omega_applied,
                    'omega_cmd': omega_applied,
                    'state': float(self.state),
                })

                if R < self.distance_threshold:
                    self.state = self.STATE_TERMINAL

            elif self.state == self.STATE_TERMINAL:
                self.publish_cmd(0.0, 0.0, 0.02)
                rospy.loginfo(
                    "========== 任务结束 ==========\n"
                    "耗时: %.3f s (signed error: %+.3f s)\n"
                    "距离误差: %.4f m\n"
                    "alpha_omega: %.3f",
                    t, t - self.T_total, R, self.alpha_omega)
                break

            self.publish_telemetry(telemetry)
            self.record_csv(t, telemetry)
            self.rate.sleep()

if __name__ == '__main__':
    try:
        ImpactTimeControlGuidance().run()
    except rospy.ROSInterruptException:
        pass