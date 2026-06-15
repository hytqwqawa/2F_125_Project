#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import select
import sys
import threading
import time

import rospy
import tf.transformations
from geometry_msgs.msg import Pose2D, PoseStamped
from std_msgs.msg import Empty, Int32
from sunray_msgs.msg import UAVControlCMD, UAVSetup


def msg_const(cls, name, fallback):
    return getattr(cls, name, fallback)


class UAVCarFollowNode:
    def __init__(self):
        rospy.init_node("uav_car_follow", anonymous=False)

        self.uav_id = rospy.get_param("~uav_id", 1)
        self.uav_name = rospy.get_param("~uav_name", "uav") + str(self.uav_id)
        self.topic_prefix = "/" + self.uav_name

        self.car_name = rospy.get_param("~car_name", "car1")
        self.car_pose_topic = rospy.get_param("~car_pose_topic", f"/{self.car_name}/pose_2d")
        self.car_pose_topic_type = rospy.get_param("~car_pose_topic_type", "pose2d").lower()
        self.follow_speed = rospy.get_param("~follow_speed", 0.5)
        self.follow_gain = rospy.get_param("~follow_gain", 0.8)
        self.arrival_thres = rospy.get_param("~arrival_thres", 0.20)
        self.return_timeout = rospy.get_param("~return_timeout", 60.0)
        self.takeoff_wait = rospy.get_param("~takeoff_wait", 10.0)
        self.hover_wait = rospy.get_param("~hover_wait", 2.0)
        self.rate_hz = rospy.get_param("~rate", 20.0)

        self.home_x = rospy.get_param("~home_x", 0.0)
        self.home_y = rospy.get_param("~home_y", 0.0)
        self.home_z = rospy.get_param("~home_z", 0.0)

        self.rate = rospy.Rate(self.rate_hz)

        self.current_pose = PoseStamped()
        self.pose_ready = False
        self.car_pose = Pose2D()
        self.car_pose_ready = False

        self.stop_requested = False
        self.cmd_id = 0

        rospy.Subscriber(self.topic_prefix + "/sunray/control_state",
                         Int32, self.uav_control_state_cb, queue_size=1)
        rospy.Subscriber(self.topic_prefix + "/sunray/stop_tutorial",
                         Empty, self.stop_tutorial_cb, queue_size=1)
        rospy.Subscriber(self.topic_prefix + "/mavros/local_position/pose",
                         PoseStamped, self.pose_cb, queue_size=10)

        if self.car_pose_topic_type == "posestamped":
            rospy.Subscriber(self.car_pose_topic,
                             PoseStamped, self.car_pose_stamped_cb, queue_size=10)
        else:
            rospy.Subscriber(self.car_pose_topic,
                             Pose2D, self.car_pose_cb, queue_size=10)

        self.control_cmd_pub = rospy.Publisher(
            self.topic_prefix + "/sunray/uav_control_cmd",
            UAVControlCMD,
            queue_size=1
        )
        self.uav_setup_pub = rospy.Publisher(
            self.topic_prefix + "/sunray/setup",
            UAVSetup,
            queue_size=1
        )

        self.CMD_TAKEOFF = msg_const(UAVControlCMD, "Takeoff", 1)
        self.CMD_HOVER = msg_const(UAVControlCMD, "Hover", 2)
        self.CMD_LAND = msg_const(UAVControlCMD, "Land", 3)
        self.CMD_XYZ_POS = msg_const(UAVControlCMD, "XYZ_POS", 4)
        self.CMD_XY_VEL_Z_POS = msg_const(UAVControlCMD, "XY_VEL_Z_POS", 5)

        self.SETUP_ARM = msg_const(UAVSetup, "ARM", 1)
        self.SETUP_SET_CONTROL_MODE = msg_const(UAVSetup, "SET_CONTROL_MODE", 4)

        rospy.loginfo("uav_car_follow started.")
        rospy.loginfo("uav_name=%s, car_name=%s", self.uav_name, self.car_name)
        rospy.loginfo("car_pose_topic=%s, car_pose_topic_type=%s", self.car_pose_topic, self.car_pose_topic_type)
        rospy.loginfo("follow_speed=%.2f m/s, home=(%.2f, %.2f, %.2f)",
                      self.follow_speed, self.home_x, self.home_y, self.home_z)
        rospy.loginfo("按 q 然后回车可退出跟随，返回 home 并降落")

    def uav_control_state_cb(self, msg):
        pass

    def stop_tutorial_cb(self, _msg):
        rospy.loginfo("/sunray/stop_tutorial received, stopping follow")
        self.stop_requested = True

    def pose_cb(self, msg):
        self.current_pose = msg
        self.pose_ready = True

    def car_pose_cb(self, msg):
        self.car_pose = msg
        self.car_pose_ready = True

    def car_pose_stamped_cb(self, msg):
        self.car_pose.x = msg.pose.position.x
        self.car_pose.y = msg.pose.position.y
        quat = (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        )
        _, _, yaw = tf.transformations.euler_from_quaternion(quat)
        self.car_pose.theta = yaw
        self.car_pose_ready = True

    @staticmethod
    def normalize_angle(a):
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    def get_xyz_yaw(self):
        p = self.current_pose.pose.position
        q = self.current_pose.pose.orientation
        _, _, yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        return p.x, p.y, p.z, yaw

    def make_cmd(self):
        cmd = UAVControlCMD()
        cmd.header.stamp = rospy.Time.now()
        cmd.cmd_id = self.cmd_id
        cmd.cmd = self.CMD_HOVER
        cmd.desired_pos = [0.0, 0.0, 0.0]
        cmd.desired_vel = [0.0, 0.0, 0.0]
        cmd.desired_acc = [0.0, 0.0, 0.0]
        cmd.desired_att = [0.0, 0.0, 0.0]
        cmd.desired_yaw = 0.0
        cmd.desired_yaw_rate = 0.0
        cmd.enable_yawRate = False
        return cmd

    def publish_cmd_once(self, cmd):
        cmd.header.stamp = rospy.Time.now()
        cmd.cmd_id = self.cmd_id
        self.cmd_id += 1
        self.control_cmd_pub.publish(cmd)

    def publish_cmd_for(self, cmd, duration):
        t0 = rospy.Time.now()
        while not rospy.is_shutdown() and (rospy.Time.now() - t0).to_sec() < duration:
            self.publish_cmd_once(cmd)
            self.rate.sleep()

    def publish_setup_for(self, setup_cmd, control_state="", duration=1.0):
        msg = UAVSetup()
        msg.cmd = setup_cmd
        msg.control_state = control_state
        t0 = rospy.Time.now()
        while not rospy.is_shutdown() and (rospy.Time.now() - t0).to_sec() < duration:
            self.uav_setup_pub.publish(msg)
            self.rate.sleep()

    def arm_and_enter_cmd_control(self):
        rospy.loginfo("arm...")
        self.publish_setup_for(self.SETUP_ARM, duration=1.0)
        rospy.sleep(0.5)
        rospy.loginfo("switch CMD_CONTROL...")
        self.publish_setup_for(self.SETUP_SET_CONTROL_MODE, "CMD_CONTROL", duration=1.0)
        rospy.sleep(0.5)

    def takeoff(self):
        rospy.loginfo("takeoff...")
        cmd = self.make_cmd()
        cmd.cmd = self.CMD_TAKEOFF
        self.publish_cmd_for(cmd, self.takeoff_wait)

    def hover(self, duration=None):
        if duration is None:
            duration = self.hover_wait
        rospy.loginfo("hover %.1f s...", duration)
        cmd = self.make_cmd()
        cmd.cmd = self.CMD_HOVER
        self.publish_cmd_for(cmd, duration)

    def land(self):
        rospy.loginfo("land...")
        cmd = self.make_cmd()
        cmd.cmd = self.CMD_LAND
        self.publish_cmd_for(cmd, 0.5)

    def wait_for_pose_ready(self, timeout=10.0):
        rospy.loginfo("wait for UAV and car pose...")
        t0 = rospy.Time.now()
        while not rospy.is_shutdown():
            if self.pose_ready and self.car_pose_ready:
                return True
            if (rospy.Time.now() - t0).to_sec() > timeout:
                rospy.logerr("timed out waiting for pose data")
                return False
            self.rate.sleep()
        return False

    def goto_point(self, x, y, z, yaw=0.0, timeout=20.0, hold_time=0.5):
        rospy.loginfo("goto point: %.2f, %.2f, %.2f", x, y, z)
        t0 = rospy.Time.now()
        while not rospy.is_shutdown():
            cmd = self.make_cmd()
            cmd.cmd = self.CMD_XYZ_POS
            cmd.desired_pos = [x, y, z]
            cmd.desired_yaw = yaw
            cmd.enable_yawRate = False
            self.publish_cmd_once(cmd)

            px, py, pz, _ = self.get_xyz_yaw()
            err = math.sqrt((px - x) ** 2 + (py - y) ** 2 + (pz - z) ** 2)
            if err < self.arrival_thres:
                rospy.loginfo("arrived at point, err=%.3f m", err)
                rospy.sleep(hold_time)
                return True
            if (rospy.Time.now() - t0).to_sec() > timeout:
                rospy.logwarn("goto timeout, err=%.3f m", err)
                return False
            self.rate.sleep()
        return False

    def follow_loop(self, z_hold):
        rospy.loginfo("start follow mode: z_hold=%.2f", z_hold)
        while not rospy.is_shutdown() and not self.stop_requested:
            if not self.car_pose_ready:
                rospy.logwarn_throttle(5, "waiting for car pose on /%s/pose_2d", self.car_name)
                self.rate.sleep()
                continue

            px, py, pz, yaw = self.get_xyz_yaw()
            dx = self.car_pose.x - px
            dy = self.car_pose.y - py
            distance = math.hypot(dx, dy)

            if distance < 0.05:
                vx = 0.0
                vy = 0.0
            else:
                target_speed = min(self.follow_speed, self.follow_gain * distance)
                vx = target_speed * dx / distance
                vy = target_speed * dy / distance

            cmd = self.make_cmd()
            cmd.cmd = self.CMD_XY_VEL_Z_POS
            cmd.desired_pos = [0.0, 0.0, z_hold]
            cmd.desired_vel = [vx, vy, 0.0]
            cmd.desired_yaw = yaw
            cmd.desired_yaw_rate = 0.0
            cmd.enable_yawRate = False
            self.publish_cmd_once(cmd)

            rospy.loginfo_throttle(
                1.0,
                "follow: car=(%.2f,%.2f), uav=(%.2f,%.2f), dist=%.2f, v=(%.2f,%.2f)",
                self.car_pose.x,
                self.car_pose.y,
                px,
                py,
                distance,
                vx,
                vy,
            )
            self.rate.sleep()

    def keyboard_listener(self):
        while not rospy.is_shutdown() and not self.stop_requested:
            try:
                if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                    line = sys.stdin.readline()
                    if line.strip().lower() == "q":
                        rospy.loginfo("keyboard q received, exiting follow mode")
                        self.stop_requested = True
                        return
            except Exception:
                time.sleep(0.1)
                continue

    def run(self):
        rospy.sleep(1.0)
        key_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        key_thread.start()

        self.arm_and_enter_cmd_control()
        self.takeoff()
        self.hover(self.hover_wait)

        if not self.wait_for_pose_ready(timeout=15.0):
            rospy.logerr("pose data missing, abort")
            return

        _, _, z_hold, _ = self.get_xyz_yaw()
        rospy.loginfo("hover height = %.2f m", z_hold)

        self.follow_loop(z_hold)

        if self.stop_requested:
            rospy.loginfo("follow stopped, returning to home and landing")
            # use current yaw so the drone doesn't keep changing yaw while descending
            _, _, _, yaw = self.get_xyz_yaw()
            self.goto_point(self.home_x, self.home_y, z_hold, yaw=yaw,
                            timeout=self.return_timeout, hold_time=1.0)
            self.land()
        else:
            rospy.loginfo("node exiting without stop request")


if __name__ == "__main__":
    try:
        UAVCarFollowNode().run()
    except rospy.ROSInterruptException:
        pass
