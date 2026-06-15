#!/usr/bin/env python3
# coding: utf-8

import rospy
import math
import tf
from geometry_msgs.msg import PoseStamped, Twist

class PNGToEnclosingGuidance:
    def __init__(self):
        rospy.init_node('png_to_enclosing_guidance', anonymous=True)
        
        # 车辆基础参数
        self.cruise_speed = 0.5
        self.max_angular = 3.0
        self.distance_threshold = 0.05  # PNG切换至Enclosing的距离阈值
        
        # 椭圆轨迹参数 (a=1.5m, b=1.0m)
        self.a = 2
        self.b = 1.0
        self.s_d = -1.0  # 环绕方向: 1为逆时针, -1为顺时针
        
        # 接入点 (Phase = pi/2, 即椭圆最上端)
        self.access_x = 0.0
        self.access_y = 1.0
        
        # 论文制导律增益
        self.nav_gain = 4.0      # PNG 导引常数 N
        self.k_r = 0.2           # 距离锁定误差增益 (需满足 0 < k_r < V)
        self.lambda_r = 1.0      # 双曲正切函数尺度
        self.k_E = 2.0           # 前置角跟踪增益
        
        # 指令滤波器状态 (用于估算 sigma_c 的导数)
        self.last_sigma_c = None
        self.filtered_sigma_c_dot = 0.0
        self.filter_alpha = 0.2  # 低通滤波系数
        
        # 状态机
        self.STATE_PNG = 1
        self.STATE_ENCLOSING = 2
        self.state = self.STATE_PNG
        
        # 自身状态
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.last_los = None     # 用于PNG阶段求视线角速率
        
        rospy.Subscriber('/vrpn_client_node/car1/pose', PoseStamped, self.pose_callback)
        self.cmd_pub = rospy.Publisher('/car1/cmd_vel', Twist, queue_size=10)
        self.rate = rospy.Rate(50)  # 50Hz dt=0.02s
        
        # 提示用户实际小车的放置要求
        rospy.loginfo("节点已初始化 | 初始阶段: PNG -> 目标点 (0, 1.0)")
        rospy.loginfo("【重要】请将小车初始放置在 (2.5, 1.0)，车头朝向正左侧 (Yaw=180度)")

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
        
    def get_ellipse_params(self, theta):
        """解析计算极坐标下椭圆的半径 rho 及其对 theta 的导数 rho' """
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        
        # 极坐标椭圆分母项
        den = math.sqrt(self.b**2 * cos_t**2 + self.a**2 * sin_t**2)
        rho = (self.a * self.b) / den
        
        # rho' 的解析求导
        den_dot_theta = (self.a**2 - self.b**2) * sin_t * cos_t / den
        rho_prime = - ((self.a * self.b) / (den**2)) * den_dot_theta
        
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
            
            # ---------------- PNG 阶段 ----------------
            if self.state == self.STATE_PNG:
                R_access = math.hypot(self.access_x - self.current_x, self.access_y - self.current_y)
                los = math.atan2(self.access_y - self.current_y, self.access_x - self.current_x)
                
                # 切换逻辑
                if R_access < self.distance_threshold:
                    self.state = self.STATE_ENCLOSING
                    rospy.loginfo("到达接入点附近，切换至 Enclosing (椭圆跟踪) 模式")
                    continue
                
                if self.last_los is None:
                    self.last_los = los
                
                los_rate = self.normalize_angle(los - self.last_los) * 50.0  # 50Hz
                self.last_los = los
                
                # 纯比例导引律
                omega = self.nav_gain * los_rate
                self.publish_cmd(V, omega)

            # ---------------- Enclosing 阶段 ----------------
            elif self.state == self.STATE_ENCLOSING:
                # 1. 计算相对靶心的极坐标 (r, theta)
                r = math.hypot(self.current_x, self.current_y)
                theta = math.atan2(self.current_y, self.current_x)
                
                # 避免原点奇异
                if r < 1e-3: r = 1e-3
                
                # 2. 论文公式计算
                rho, rho_prime = self.get_ellipse_params(theta)
                
                c_val = rho_prime / r
                A_rho = math.sqrt(1.0 + c_val**2)
                phi = math.atan(c_val)
                
                # 距离锁定误差 (Eq. 7)
                e_r = r - rho
                Psi = self.k_r * math.tanh(self.lambda_r * e_r)
                
                # 基础前置角指令 (Eq. 10)
                # 保护 arccos 域防止过饱和
                arccos_input = max(-0.999, min(0.999, Psi / (V * A_rho)))
                sigma_c = -phi + self.s_d * math.acos(arccos_input)
                
                # 实际前置角 (基于论文 Section II.C 定义: sigma_E = gamma - theta - pi)
                sigma = self.normalize_angle(self.current_yaw - theta - math.pi)
                e_sigma = self.normalize_angle(sigma - sigma_c)
                
                # 3. 指令微分与滤波 (Section V.B)
                dt = 0.02
                if self.last_sigma_c is None:
                    self.last_sigma_c = sigma_c
                    raw_sigma_c_dot = 0.0
                else:
                    raw_sigma_c_dot = self.normalize_angle(sigma_c - self.last_sigma_c) / dt
                    
                self.last_sigma_c = sigma_c
                self.filtered_sigma_c_dot = self.filter_alpha * raw_sigma_c_dot + (1.0 - self.filter_alpha) * self.filtered_sigma_c_dot
                
                # 4. 侧向加速度指令计算 (Eq. 11)
                A_E = V * (self.filtered_sigma_c_dot - (V/r) * math.sin(sigma) - self.k_E * e_sigma)
                
                # 转换为角速度
                omega = A_E / V
                self.publish_cmd(V, omega)

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
        guidance = PNGToEnclosingGuidance()
        guidance.run()
    except rospy.ROSInterruptException:
        pass