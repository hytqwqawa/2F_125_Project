#!/usr/bin/env python3
# coding: utf-8

import rospy
import math
import tf
import csv
import os
import bisect
from datetime import datetime

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64


class ImpactTimeControlGuidance:
    def __init__(self):
        rospy.init_node('impact_time_control_guidance', anonymous=True)

        # ================= 任务参数 =================
        self.target_x = -1.5
        self.target_y = 0.5

        self.cruise_speed = 0.4
        self.terminal_speed = 0.2
        self.desired_impact_time = 20.0
        self.lambda_desired = math.radians(135.0)

        # ================= 平台约束 =================
        self.v_min = 0.05
        self.v_max = 0.50
        self.a_f = 0.5
        self.a_b = 0.5
        self.Delta_L_max = 5.0

        self.distance_budget_factor = 1.65
        self.max_angular = 3.0
        self.max_yaw_accel = 2.0
        self.accel_duration = 0.2
        self.distance_threshold = 0.05

        # ================= Dubins + tracking 参数 =================
        self.turning_radius = 1       # Dubins 最小转弯半径，可按场地调 0.45~0.80
        self.path_resolution = 0.02      # 路径采样间隔

        self.lookahead_base = 0.25
        self.lookahead_gain = 0.75
        self.lookahead_min = 0.30
        self.lookahead_max = 0.80

        self.heading_hold_gain = 2.0
        self.terminal_blind_R = 0.15
        self.terminal_blind_tgo = 0.5

        self.omega_filter_alpha = 0.15
        self.last_cmd_omega = 0.0
        self.filtered_omega = 0.0

        # ================= 状态变量 =================
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.actual_speed = 0.0

        self.path_x = []
        self.path_y = []
        self.path_yaw = []
        self.path_s = []
        self.path_len = 0.0
        self.path_mode = ""
        self.path_params = []
        self.nearest_idx_last = 0

        self.c_coeff = 0.0
        self.locked_yaw = self.lambda_desired

        self.STATE_ACCEL = 1
        self.STATE_TRACK = 2
        self.STATE_HOLD = 3
        self.STATE_TERMINAL = 4
        self.state = self.STATE_ACCEL
        self.last_time_sec = None

        # ================= CSV 记录 =================
        log_dir = os.path.expanduser('~')
        filename = datetime.now().strftime("dubins_tracking_data_%Y%m%d_%H%M%S.csv")
        self.csv_path = os.path.join(log_dir, filename)
        self.csv_file = open(self.csv_path, 'w')
        self.csv_writer = csv.writer(self.csv_file)

        # [修改] 按要求插入 pos_x, pos_y 保证前15个字段一致
        self.csv_keys = [
            'relative_dist', 'tgo', 'speed_actual', 'speed_desired',
            'sigma_actual', 'sigma_desired', 'lambda_actual', 'lambda_error',
            'yaw_actual', 'angular_vel', 'sigma_lambda', 'sigma_dot',
            'los_rate_raw', 'omega_raw', 'omega_cmd',
            'pos_x', 'pos_y',  # <==== 插入点
            'path_lateral_error', 'path_s_ref', 'path_s_near',
            'curvature_cmd', 'path_length'
        ]

        headers = ['time'] + [
            k + '_deg' if any(x in k for x in ['sigma', 'lambda', 'yaw']) else k
            for k in self.csv_keys
        ]
        self.csv_writer.writerow(headers)
        rospy.on_shutdown(self.shutdown_hook)

        # ================= ROS 接口 =================
        rospy.Subscriber('/vrpn_client_node/car1/pose', PoseStamped, self.pose_callback)
        rospy.Subscriber('/car1/odom', Odometry, self.odom_callback)
        self.cmd_pub = rospy.Publisher('/car1/cmd_vel', Twist, queue_size=10)

        topics = [
            'angular_vel', 'sigma_actual', 'sigma_desired',
            'lambda_actual', 'lambda_error', 'relative_dist', 'tgo',
            'speed_desired', 'speed_actual', 'sigma_lambda',
            'sigma_dot', 'los_rate_raw', 'omega_raw', 'omega_cmd',
            'yaw_actual', 'path_lateral_error', 'path_s_ref',
            'path_s_near', 'curvature_cmd', 'path_length'
        ]

        self.pubs = {
            name: rospy.Publisher('/itacg/{}'.format(name), Float64, queue_size=10)
            for name in topics
        }

        self.rate = rospy.Rate(50)
        rospy.loginfo("Dubins + time-scaling + Pure Pursuit baseline initialized | CSV enabled")

    # ================= 基础工具 =================

    def shutdown_hook(self):
        if not self.csv_file.closed:
            self.csv_file.close()
            rospy.loginfo("CSV数据已保存至: %s", self.csv_path)

    def record_csv(self, t, telemetry):
        row = [t]
        for key in self.csv_keys:
            val = telemetry.get(key, 0.0)
            if any(x in key for x in ['sigma', 'lambda', 'yaw']):
                val = math.degrees(val)
            row.append(val)
        self.csv_writer.writerow(row)

    def publish_telemetry(self, data):
        for key, val in data.items():
            if key in self.pubs:
                if any(x in key for x in ['sigma', 'lambda', 'yaw']):
                    self.pubs[key].publish(math.degrees(val))
                else:
                    self.pubs[key].publish(val)

    def sat(self, x, xmin, xmax):
        return max(xmin, min(xmax, x))

    def slew_limit(self, x, x_last, rate, dt):
        return x_last + self.sat(x - x_last, -rate * dt, rate * dt)

    def normalize_angle(self, angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def mod2pi(self, angle):
        return angle % (2.0 * math.pi)

    def get_line_of_sight_angle(self):
        return math.atan2(self.target_y - self.current_y,
                          self.target_x - self.current_x)

    def odom_callback(self, msg):
        self.actual_speed = math.hypot(
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y
        )

    def pose_callback(self, msg):
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y
        q = msg.pose.orientation
        _, _, self.current_yaw = tf.transformations.euler_from_quaternion(
            (q.x, q.y, q.z, q.w)
        )

    def publish_cmd(self, v, omega, dt):
        omega = self.slew_limit(omega, self.last_cmd_omega,
                                self.max_yaw_accel, dt)

        cmd = Twist()
        cmd.linear.x = self.sat(v, 0.0, self.v_max)
        cmd.angular.z = self.sat(omega, -self.max_angular, self.max_angular)

        self.last_cmd_omega = cmd.angular.z
        self.cmd_pub.publish(cmd)

    # ================= 速度可行距离 =================

    def calc_S_max(self, v0, vg, td):
        t_acc = (self.v_max - v0) / self.a_f
        t_dec = (self.v_max - vg) / self.a_b

        if t_acc + t_dec <= td:
            t_cr = td - t_acc - t_dec
            return ((v0 + self.v_max) * t_acc / 2.0
                    + self.v_max * t_cr
                    + (self.v_max + vg) * t_dec / 2.0)

        t1 = (vg - v0 + self.a_b * td) / (self.a_f + self.a_b)
        vp = v0 + self.a_f * t1
        return (v0 + vp) * t1 / 2.0 + (vp + vg) * (td - t1) / 2.0

    def calc_S_min(self, v0, vg, td):
        t_dec = (v0 - self.v_min) / self.a_b
        t_acc = (vg - self.v_min) / self.a_f

        if t_dec + t_acc <= td:
            t_cr = td - t_dec - t_acc
            return ((v0 + self.v_min) * t_dec / 2.0
                    + self.v_min * t_cr
                    + (self.v_min + vg) * t_acc / 2.0)

        t1 = (v0 - vg + self.a_f * td) / (self.a_b + self.a_f)
        vv = v0 - self.a_b * t1
        return (v0 + vv) * t1 / 2.0 + (vv + vg) * (td - t1) / 2.0

    # ================= Dubins 候选路径 =================

    def dubins_LSL(self, alpha, beta, d):
        sa, sb = math.sin(alpha), math.sin(beta)
        ca, cb = math.cos(alpha), math.cos(beta)
        cab = math.cos(alpha - beta)

        tmp = d + sa - sb
        p2 = 2.0 + d * d - 2.0 * cab + 2.0 * d * (sa - sb)
        if p2 < 0.0:
            return None

        p = math.sqrt(p2)
        t = self.mod2pi(-alpha + math.atan2(cb - ca, tmp))
        q = self.mod2pi(beta - math.atan2(cb - ca, tmp))
        return t, p, q, "LSL"

    def dubins_RSR(self, alpha, beta, d):
        sa, sb = math.sin(alpha), math.sin(beta)
        ca, cb = math.cos(alpha), math.cos(beta)
        cab = math.cos(alpha - beta)

        tmp = d - sa + sb
        p2 = 2.0 + d * d - 2.0 * cab + 2.0 * d * (-sa + sb)
        if p2 < 0.0:
            return None

        p = math.sqrt(p2)
        t = self.mod2pi(alpha - math.atan2(ca - cb, tmp))
        q = self.mod2pi(-beta + math.atan2(ca - cb, tmp))
        return t, p, q, "RSR"

    def dubins_LSR(self, alpha, beta, d):
        sa, sb = math.sin(alpha), math.sin(beta)
        ca, cb = math.cos(alpha), math.cos(beta)
        cab = math.cos(alpha - beta)

        p2 = -2.0 + d * d + 2.0 * cab + 2.0 * d * (sa + sb)
        if p2 < 0.0:
            return None

        p = math.sqrt(p2)
        tmp = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p)
        t = self.mod2pi(-alpha + tmp)
        q = self.mod2pi(-beta + tmp)
        return t, p, q, "LSR"

    def dubins_RSL(self, alpha, beta, d):
        sa, sb = math.sin(alpha), math.sin(beta)
        ca, cb = math.cos(alpha), math.cos(beta)
        cab = math.cos(alpha - beta)

        p2 = -2.0 + d * d + 2.0 * cab - 2.0 * d * (sa + sb)
        if p2 < 0.0:
            return None

        p = math.sqrt(p2)
        tmp = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, p)
        t = self.mod2pi(alpha - tmp)
        q = self.mod2pi(beta - tmp)
        return t, p, q, "RSL"

    def dubins_RLR(self, alpha, beta, d):
        sa, sb = math.sin(alpha), math.sin(beta)
        ca, cb = math.cos(alpha), math.cos(beta)
        cab = math.cos(alpha - beta)

        tmp = (6.0 - d * d + 2.0 * cab + 2.0 * d * (sa - sb)) / 8.0
        if abs(tmp) > 1.0:
            return None

        p = self.mod2pi(2.0 * math.pi - math.acos(tmp))
        t = self.mod2pi(alpha - math.atan2(ca - cb, d - sa + sb) + p / 2.0)
        q = self.mod2pi(alpha - beta - t + p)
        return t, p, q, "RLR"

    def dubins_LRL(self, alpha, beta, d):
        sa, sb = math.sin(alpha), math.sin(beta)
        ca, cb = math.cos(alpha), math.cos(beta)
        cab = math.cos(alpha - beta)

        tmp = (6.0 - d * d + 2.0 * cab + 2.0 * d * (-sa + sb)) / 8.0
        if abs(tmp) > 1.0:
            return None

        p = self.mod2pi(2.0 * math.pi - math.acos(tmp))
        t = self.mod2pi(-alpha - math.atan2(ca - cb, d + sa - sb) + p / 2.0)
        q = self.mod2pi(beta - alpha - t + p)
        return t, p, q, "LRL"

    def shortest_dubins(self, x0, y0, yaw0, xg, yg, yawg):
        dx = xg - x0
        dy = yg - y0
        D = math.hypot(dx, dy)
        rho = self.turning_radius

        if D < 1e-6:
            rospy.logwarn("起点与终点过近，Dubins 路径可能退化")
            D = 1e-6

        theta = math.atan2(dy, dx)
        d = D / rho

        alpha = self.mod2pi(yaw0 - theta)
        beta = self.mod2pi(yawg - theta)

        candidates = []
        for fn in [self.dubins_LSL, self.dubins_RSR, self.dubins_LSR,
                   self.dubins_RSL, self.dubins_RLR, self.dubins_LRL]:
            out = fn(alpha, beta, d)
            if out is not None:
                t, p, q, mode = out
                length = (t + p + q) * rho
                candidates.append((length, [t, p, q], mode))

        if not candidates:
            return None

        candidates.sort(key=lambda z: z[0])
        return candidates[0]

    def append_point(self, xs, ys, yaws, ss, x, y, yaw, s):
        if len(xs) == 0 or math.hypot(x - xs[-1], y - ys[-1]) > 1e-6:
            xs.append(x)
            ys.append(y)
            yaws.append(self.normalize_angle(yaw))
            ss.append(s)

    def sample_segment(self, xs, ys, yaws, ss, x, y, yaw, mode, length, s_now):
        rho = self.turning_radius
        ds = self.path_resolution
        traveled = 0.0

        while traveled < length - 1e-9:
            step = min(ds, length - traveled)

            if mode == 'S':
                x += step * math.cos(yaw)
                y += step * math.sin(yaw)

            else:
                sign = 1.0 if mode == 'L' else -1.0
                yaw_new = yaw + sign * step / rho
                x += sign * rho * (math.sin(yaw_new) - math.sin(yaw))
                y += sign * rho * (math.cos(yaw) - math.cos(yaw_new))
                yaw = yaw_new

            traveled += step
            s_now += step
            self.append_point(xs, ys, yaws, ss, x, y, yaw, s_now)

        return x, y, yaw, s_now

    def generate_dubins_path(self):
        x0, y0, yaw0 = self.current_x, self.current_y, self.current_yaw
        xg, yg, yawg = self.target_x, self.target_y, self.lambda_desired

        best = self.shortest_dubins(x0, y0, yaw0, xg, yg, yawg)
        if best is None:
            rospy.logerr("Dubins 规划失败")
            return False

        length, params, mode = best
        xs, ys, yaws, ss = [], [], [], []
        x, y, yaw, s_now = x0, y0, yaw0, 0.0

        self.append_point(xs, ys, yaws, ss, x, y, yaw, s_now)

        for m, param in zip(mode, params):
            seg_len = param * self.turning_radius
            x, y, yaw, s_now = self.sample_segment(
                xs, ys, yaws, ss, x, y, yaw, m, seg_len, s_now
            )

        # 强制末端对齐，避免采样误差影响终端判据
        xs[-1], ys[-1], yaws[-1], ss[-1] = xg, yg, yawg, length

        self.path_x = xs
        self.path_y = ys
        self.path_yaw = yaws
        self.path_s = ss
        self.path_len = length
        self.path_mode = mode
        self.path_params = params

        return True

    def plan_path(self):
        if not self.generate_dubins_path():
            return False

        S_min = self.calc_S_min(self.cruise_speed,
                                self.terminal_speed,
                                self.desired_impact_time)
        S_max = self.calc_S_max(self.cruise_speed,
                                self.terminal_speed,
                                self.desired_impact_time)

        R0 = math.hypot(self.target_x - self.current_x,
                        self.target_y - self.current_y)
        L_target = R0 * self.distance_budget_factor

        self.c_coeff = 30.0 * (
            self.path_len / self.desired_impact_time
            - 0.5 * (self.cruise_speed + self.terminal_speed)
        )

        rospy.loginfo(
            "Dubins planned | mode=%s | rho=%.3f | R0=%.3f | L_target=%.3f | L_dubins=%.3f | feasible=[%.3f, %.3f] | c=%.3f",
            self.path_mode, self.turning_radius, R0, L_target,
            self.path_len, S_min, S_max, self.c_coeff
        )

        if not (S_min <= self.path_len <= S_max):
            rospy.logwarn(
                "Dubins路径长度不在速度可行区间内: L=%.3f, [Smin,Smax]=[%.3f,%.3f]。"
                "该工况可能体现 path-first baseline 的时间化局限。",
                self.path_len, S_min, S_max
            )

        return True

    # ================= 时间缩放 =================

    def speed_and_reference_s(self, t):
        s = self.sat(t / self.desired_impact_time, 0.0, 1.0)

        b = 3.0 * s ** 2 - 2.0 * s ** 3
        bc = s ** 2 * (1.0 - s) ** 2

        V = (self.cruise_speed
             + (self.terminal_speed - self.cruise_speed) * b
             + self.c_coeff * bc)

        F_s = (self.cruise_speed * s
               + (self.terminal_speed - self.cruise_speed)
               * (s ** 3 - 0.5 * s ** 4)
               + self.c_coeff
               * (s ** 3 / 3.0 - 0.5 * s ** 4 + s ** 5 / 5.0))

        s_ref = self.desired_impact_time * F_s
        return self.sat(V, self.v_min, self.v_max), self.sat(s_ref, 0.0, self.path_len)

    # ================= 路径查询与 Pure Pursuit =================

    def path_at_s(self, s_query):
        s_query = self.sat(s_query, 0.0, self.path_len)
        idx = bisect.bisect_left(self.path_s, s_query)

        if idx <= 0:
            return self.path_x[0], self.path_y[0], self.path_yaw[0]
        if idx >= len(self.path_s):
            return self.path_x[-1], self.path_y[-1], self.path_yaw[-1]

        s0, s1 = self.path_s[idx - 1], self.path_s[idx]
        r = 0.0 if s1 <= s0 else (s_query - s0) / (s1 - s0)

        x = self.path_x[idx - 1] + r * (self.path_x[idx] - self.path_x[idx - 1])
        y = self.path_y[idx - 1] + r * (self.path_y[idx] - self.path_y[idx - 1])
        yaw = self.path_yaw[idx - 1]

        return x, y, yaw

    def nearest_path_index(self):
        start = max(0, self.nearest_idx_last - 40)
        end = min(len(self.path_x), self.nearest_idx_last + 220)

        best_i = start
        best_d2 = 1e18

        for i in range(start, end):
            dx = self.current_x - self.path_x[i]
            dy = self.current_y - self.path_y[i]
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_i = i

        self.nearest_idx_last = best_i
        return best_i, math.sqrt(best_d2)

    def pure_pursuit_control(self, V_cmd):
        idx, _ = self.nearest_path_index()
        s_near = self.path_s[idx]

        Ld = self.sat(self.lookahead_base + self.lookahead_gain * V_cmd,
                      self.lookahead_min, self.lookahead_max)

        s_look = min(self.path_len, s_near + Ld)
        xL, yL, yaw_ref = self.path_at_s(s_look)

        bearing = math.atan2(yL - self.current_y, xL - self.current_x)
        alpha = self.normalize_angle(bearing - self.current_yaw)

        Ld_actual = max(0.15, math.hypot(xL - self.current_x, yL - self.current_y))
        kappa = 2.0 * math.sin(alpha) / Ld_actual
        omega_raw = V_cmd * kappa

        # Dubins 在段切换处曲率不连续，终段加一点航向保持有利于稳定过门
        remain_s = self.path_len - s_near
        if remain_s < 0.80:
            w = self.sat((0.80 - remain_s) / 0.80, 0.0, 1.0)
            heading_err = self.normalize_angle(self.lambda_desired - self.current_yaw)
            omega_raw += w * self.heading_hold_gain * heading_err

        self.filtered_omega = (
            self.omega_filter_alpha * omega_raw
            + (1.0 - self.omega_filter_alpha) * self.filtered_omega
        )

        xN, yN, yawN = self.path_x[idx], self.path_y[idx], self.path_yaw[idx]
        dx = self.current_x - xN
        dy = self.current_y - yN
        e_y = -math.sin(yawN) * dx + math.cos(yawN) * dy

        return self.filtered_omega, omega_raw, kappa, alpha, e_y, s_near, yaw_ref, bearing

    # ================= 主循环 =================

    def run(self):
        rospy.loginfo("等待动捕数据...")
        try:
            msg = rospy.wait_for_message(
                '/vrpn_client_node/car1/pose',
                PoseStamped,
                timeout=5.0
            )
            self.pose_callback(msg)
        except rospy.ROSException:
            return rospy.logerr("未收到动捕数据，节点退出")

        if not self.plan_path():
            return

        start_time = rospy.Time.now()

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            t = (now - start_time).to_sec()

            if self.last_time_sec is None:
                dt_real = 0.02
            else:
                dt_real = now.to_sec() - self.last_time_sec
                if dt_real <= 0.001:
                    dt_real = 0.02
            self.last_time_sec = now.to_sec()

            t_go = self.desired_impact_time - t
            R = math.hypot(self.target_x - self.current_x,
                           self.target_y - self.current_y)

            los = self.get_line_of_sight_angle()
            sigma = self.normalize_angle(self.current_yaw - los)
            lambda_e = self.normalize_angle(los - self.lambda_desired)

            telemetry = {
                'relative_dist': R,
                'tgo': t_go,
                'speed_actual': self.actual_speed,
                'sigma_actual': sigma,
                'lambda_actual': los,
                'lambda_error': lambda_e,
                'yaw_actual': self.current_yaw,
                'pos_x': self.current_x,       # [新增] 压入当前 X 坐标
                'pos_y': self.current_y,       # [新增] 压入当前 Y 坐标
                'path_length': self.path_len
            }

            # ---------- 加速段 ----------
            if self.state == self.STATE_ACCEL:
                if t >= self.accel_duration:
                    self.state = self.STATE_TRACK
                    rospy.loginfo("进入 Dubins path tracking 阶段")
                    continue

                V_cmd = self.cruise_speed * (t / self.accel_duration)
                self.publish_cmd(V_cmd, 0.0, dt_real)

                telemetry.update({
                    'speed_desired': V_cmd,
                    'sigma_desired': 0.0,
                    'angular_vel': 0.0,
                    'omega_cmd': self.last_cmd_omega
                })

                self.publish_telemetry(telemetry)
                self.record_csv(t, telemetry)

            # ---------- Dubins 路径跟踪 ----------
            elif self.state == self.STATE_TRACK:
                V_cmd, s_ref = self.speed_and_reference_s(t)

                omega, omega_raw, kappa, alpha, e_y, s_near, yaw_ref, bearing = \
                    self.pure_pursuit_control(V_cmd)

                self.publish_cmd(V_cmd, omega, dt_real)

                telemetry.update({
                    'speed_desired': V_cmd,
                    'sigma_desired': self.normalize_angle(bearing - los),
                    'angular_vel': omega,
                    'sigma_lambda': alpha,
                    'sigma_dot': self.normalize_angle(yaw_ref - self.current_yaw),
                    'los_rate_raw': kappa,
                    'omega_raw': omega_raw,
                    'omega_cmd': self.last_cmd_omega,
                    'path_lateral_error': e_y,
                    'path_s_ref': s_ref,
                    'path_s_near': s_near,
                    'curvature_cmd': kappa
                })

                self.publish_telemetry(telemetry)
                self.record_csv(t, telemetry)

                rospy.loginfo_throttle(
                    0.5,
                    "DUBINS | mode=%s | R=%.3f | e_y=%.3f | s=%.2f/%.2f | alpha=%.1fdeg | V=%.2f | w=%.2f",
                    self.path_mode, R, e_y, s_near, self.path_len,
                    math.degrees(alpha), V_cmd, self.last_cmd_omega
                )

                if R < self.terminal_blind_R or t_go <= self.terminal_blind_tgo:
                    self.state = self.STATE_HOLD
                    self.locked_yaw = self.lambda_desired
                    self.filtered_omega = 0.0
                    rospy.loginfo("切入终端航向保持模式")
                    continue

            # ---------- 终端航向保持 ----------
            elif self.state == self.STATE_HOLD:
                V_cmd, s_ref = self.speed_and_reference_s(t)

                yaw_error = self.normalize_angle(self.locked_yaw - self.current_yaw)
                omega_raw = self.heading_hold_gain * yaw_error

                self.filtered_omega = (
                    self.omega_filter_alpha * omega_raw
                    + (1.0 - self.omega_filter_alpha) * self.filtered_omega
                )

                self.publish_cmd(V_cmd, self.filtered_omega, dt_real)

                telemetry.update({
                    'speed_desired': V_cmd,
                    'sigma_desired': self.normalize_angle(self.locked_yaw - los),
                    'angular_vel': self.filtered_omega,
                    'sigma_lambda': yaw_error,
                    'sigma_dot': yaw_error,
                    'los_rate_raw': 0.0,
                    'omega_raw': omega_raw,
                    'omega_cmd': self.last_cmd_omega,
                    'path_s_ref': s_ref,
                    'path_s_near': self.path_len,
                    'curvature_cmd': 0.0
                })

                self.publish_telemetry(telemetry)
                self.record_csv(t, telemetry)

                if R < self.distance_threshold:
                    self.state = self.STATE_TERMINAL

            # ---------- 停车 ----------
            elif self.state == self.STATE_TERMINAL:
                self.publish_cmd(0.0, 0.0, 0.02)
                rospy.loginfo(
                    "========== Dubins path tracking task finished ==========\n"
                    "耗时: %.2fs | 时间误差: %.3fs | 距离误差: %.3fm | 航向误差: %.2fdeg",
                    t,
                    abs(self.desired_impact_time - t),
                    R,
                    math.degrees(self.normalize_angle(self.current_yaw - self.lambda_desired))
                )
                break

            self.rate.sleep()


if __name__ == '__main__':
    try:
        ImpactTimeControlGuidance().run()
    except rospy.ROSInterruptException:
        pass