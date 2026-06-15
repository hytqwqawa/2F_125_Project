#!/usr/bin/env python3
# coding: utf-8

import rospy
import math
import tf
from geometry_msgs.msg import PoseStamped, Pose2D

def pose_callback(msg, args):
    pub_rad, pub_deg, car_name = args
    
    # 提取位置
    x = msg.pose.position.x
    y = msg.pose.position.y
    
    # 四元数转欧拉角，获取偏航角（弧度）
    quat = (
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
        msg.pose.orientation.w
    )
    _, _, yaw_rad = tf.transformations.euler_from_quaternion(quat)
    
    # 转换为度
    yaw_deg = yaw_rad * 180.0 / math.pi
    
    # 发布弧度版本的 Pose2D（符合 ROS 规范）
    pose2d_rad = Pose2D()
    pose2d_rad.x = x
    pose2d_rad.y = y
    pose2d_rad.theta = yaw_rad
    pub_rad.publish(pose2d_rad)
    
    # 发布度数版本的 Pose2D（非标准，仅用于直观观察）
    pose2d_deg = Pose2D()
    pose2d_deg.x = x
    pose2d_deg.y = y
    pose2d_deg.theta = yaw_deg
    pub_deg.publish(pose2d_deg)
    
    # 打印日志：显示度数（带有车辆名称前缀）
    rospy.loginfo("[%s] 当前位置: x=%.2f m, y=%.2f m, yaw=%.2f deg", car_name, x, y, yaw_deg)

def main():
    rospy.init_node('mocap_subscriber_multi', anonymous=True)
    
    # 获取参数：读取多辆车的名称字符串，默认值为 'car1'
    car_names_str = rospy.get_param('~car_names', 'car1')
    
    # 解析字符串为列表，并去除可能的空格
    car_names = [name.strip() for name in car_names_str.split(',') if name.strip()]
    
    rospy.loginfo("开始监听以下车辆动捕数据: %s", car_names)
    
    # 使用列表保存所有的订阅器对象，防止在作用域结束时被回收
    subscribers = []
    
    # 遍历车辆列表，为每一辆车创建独立的话题收发逻辑
    for car in car_names:
        pose_topic = f"/vrpn_client_node/{car}/pose"
        rad_topic  = f"/{car}/pose_2d"
        deg_topic  = f"/{car}/pose_deg"
        
        # 实例化对应车辆的发布器
        pub_rad = rospy.Publisher(rad_topic, Pose2D, queue_size=10)
        pub_deg = rospy.Publisher(deg_topic, Pose2D, queue_size=10)
        
        # 订阅动捕位姿，利用 callback_args 将该车专属的发布器及名称传递给同一个回调函数
        sub = rospy.Subscriber(pose_topic, PoseStamped, pose_callback, (pub_rad, pub_deg, car))
        subscribers.append(sub)
        
        rospy.loginfo(" - [%s] 话题已映射: %s -> %s & %s", car, pose_topic, rad_topic, deg_topic)
    
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass