#!/usr/bin/env python3
# coding: utf-8

import rospy
import math
import tf
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Float64          # 新增：发布浮点数据

class PNGToRoseEnclosingGuidance:
    def __init__(self):
        rospy.init_node('png_to_enclosing_guidance', anonymous=True)
        
        # 车辆基础参数
        self.cruise_speed = 0.3
        self.max_angular = 2.0           # 依要求限制
        self.distance_threshold = 0.05   # PNG切换至Enclosing的距离阈值
        
        # 三瓣形(Rose Curve)轨迹参数: rho(theta) = R0 + R1 * cos(k * theta)
        self.R0 = 1.3
        self.R1 = 0.3
        self.k = 3.0
        self.s_d = -1.0  # 环绕方向: -1为逆时针 (随极角增加方向)
        
        # 接入点 (Phase = 0, 右侧瓣尖)
        self.access_x = 1.6
        self.access_y = 0.0
        
        # 制导律增益
        self.nav_gain = 3.0      # PNG 导引常数
        self.k_r = 0.2           # 距离锁定误差增益 (0 < k_r < V)
        self.lambda_r = 1.0      # 双曲正切函数尺度
        self.k_E = 2.0           # 前置角跟踪增益
        
        # 指令滤波器状态
        self.last_sigma_c = None
        self.filtered_sigma_c_dot = 0.0
        self.filter_alpha = 0.2
        
        # 状态机
        self.STATE_PNG = 1
        self.STATE_ENCLOSING = 2
        self.state = self.STATE_PNG
        
        # 自身状态
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.last_los = None
        
        rospy.Subscriber('/vrpn_client_node/car1/pose', PoseStamped, self.pose_callback)
        self.cmd_pub = rospy.Publisher('/car1/cmd_vel', Twist, queue_size=10)
        
        # ------------------ 新增发布器 ------------------
        self.pub_r = rospy.Publisher('/enclosing_r', Float64, queue_size=10)        # 距离R (米)
        self.pub_acc = rospy.Publisher('/enclosing_acc', Float64, queue_size=10)    # 法向加速度 A = ω * V (m/s²)
        self.pub_theta_deg = rospy.Publisher('/enclosing_theta_deg', Float64, queue_size=10)  # 极角 (度)
        self.pub_sigma_deg = rospy.Publisher('/enclosing_sigma_deg', Float64, queue_size=10)  # 前置角 (度)
        # ----------------------------------------------
        
        self.rate = rospy.Rate(50)  # 50Hz dt=0.02s
        
        rospy.loginfo("节点初始化 | 初始阶段: PNG -> 目标接入点 (1.6, 0.0)")
        rospy.loginfo("【部署提示】请将小车放置在 (1.6, -1.5) 附近，车头朝向正上方 (Yaw = 90度) 以保障完美相切。")

    def pose_callback(self, msg):
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y
        quat = (msg.pose.orientation.x, msg.pose.orientation.y,
                msg.pose.orientation.z, msg.pose.orientation.w)
        _, _, self.current_yaw = tf.transformations.euler_from_quaternion(quat)

    def normalize_angle(self, angle):
        while angle > math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle
        
    def get_shape_params(self, theta):
        """解析计算极坐标下瓣形的半径 rho 及其对 theta 的导数 rho' """
        rho = self.R0 + self.R1 * math.cos(self.k * theta)
        rho_prime = -self.R1 * self.k * math.sin(self.k * theta)
        return rho, rho_prime

    def run(self):
        rospy.loginfo("等待动捕数据...")
        try:
            rospy.wait_for_message('/vrpn_client_node/car1/pose', PoseStamped, timeout=5.0)
        except rospy.ROSException:
            rospy.logerr("未收到动捕数据，节点退出。")
            return
            
        while not rospy.is_shutdown():
            V = self.cruise_speed
            # 计算通用量：极径 r、极角 theta（弧度）
            r = math.hypot(self.current_x, self.current_y)
            theta = math.atan2(self.current_y, self.current_x)
            # 前置角 sigma 定义（与代码中 Enclosing 阶段一致）：sigma = ψ - θ - π
            sigma = self.normalize_angle(self.current_yaw - theta - math.pi)
            
            # ---------------- PNG 阶段 ----------------
            if self.state == self.STATE_PNG:
                R_access = math.hypot(self.access_x - self.current_x, self.access_y - self.current_y)
                los = math.atan2(self.access_y - self.current_y, self.access_x - self.current_x)
                
                if R_access < self.distance_threshold:
                    self.state = self.STATE_ENCLOSING
                    rospy.loginfo("到达接入点，切换至 Enclosing (瓣形轨迹跟踪) 模式")
                    continue
                
                if self.last_los is None:
                    self.last_los = los
                
                los_rate = self.normalize_angle(los - self.last_los) * 50.0  # 50Hz
                self.last_los = los
                
                omega = self.nav_gain * los_rate
                self.publish_cmd(V, omega)
                
                # 发布通用数据（PNG 阶段）
                self.pub_r.publish(r)
                A = omega * V
                self.pub_acc.publish(A)
                theta_deg = theta * 180.0 / math.pi
                self.pub_theta_deg.publish(theta_deg)
                sigma_deg = sigma * 180.0 / math.pi
                self.pub_sigma_deg.publish(sigma_deg)

            # ---------------- Enclosing 阶段 ----------------
            elif self.state == self.STATE_ENCLOSING:
                # r, theta, sigma 已在上方计算，直接使用
                if r < 1e-3: r = 1e-3
                
                # 提取瓣形几何状态参数
                rho, rho_prime = self.get_shape_params(theta)
                
                c_val = rho_prime / r
                A_rho = math.sqrt(1.0 + c_val**2)
                phi = math.atan(c_val)
                
                e_r = r - rho
                Psi = self.k_r * math.tanh(self.lambda_r * e_r)
                
                arccos_input = max(-0.999, min(0.999, Psi / (V * A_rho)))
                sigma_c = -phi + self.s_d * math.acos(arccos_input)
                
                e_sigma = self.normalize_angle(sigma - sigma_c)
                
                dt = 0.02
                if self.last_sigma_c is None:
                    self.last_sigma_c = sigma_c
                    raw_sigma_c_dot = 0.0
                else:
                    raw_sigma_c_dot = self.normalize_angle(sigma_c - self.last_sigma_c) / dt
                    
                self.last_sigma_c = sigma_c
                self.filtered_sigma_c_dot = self.filter_alpha * raw_sigma_c_dot + (1.0 - self.filter_alpha) * self.filtered_sigma_c_dot
                
                A_E = V * (self.filtered_sigma_c_dot - (V/r) * math.sin(sigma) - self.k_E * e_sigma)
                omega = A_E / V
                
                self.publish_cmd(V, omega)
                
                # 发布通用数据（Enclosing 阶段）
                self.pub_r.publish(r)
                self.pub_acc.publish(A_E)          # 法向加速度直接使用 A_E
                theta_deg = theta * 180.0 / math.pi
                self.pub_theta_deg.publish(theta_deg)
                sigma_deg = sigma * 180.0 / math.pi
                self.pub_sigma_deg.publish(sigma_deg)

            self.rate.sleep()

    def publish_cmd(self, v, omega):
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = max(-self.max_angular, min(self.max_angular, omega))
        self.cmd_pub.publish(cmd)
        
    def stop(self):
        self.cmd_pub.publish(Twist())

if __name__ == '__main__':
    try:
        guidance = PNGToRoseEnclosingGuidance()
        guidance.run()
    except rospy.ROSInterruptException:
        pass