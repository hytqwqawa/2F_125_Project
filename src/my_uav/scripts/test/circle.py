#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy
import tf.transformations

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty, Int32
from sunray_msgs.msg import UAVState, UAVControlCMD, UAVSetup


def msg_const(cls, name, fallback):
    return getattr(cls, name, fallback)


class CircleHeightHoldNode:
    def __init__(self):
        rospy.init_node("circle_height_hold", anonymous=False)

        self.uav_id = rospy.get_param("~uav_id", 1)
        self.uav_name = rospy.get_param("~uav_name", "uav") + str(self.uav_id)  
        self.topic_prefix = "/" + self.uav_name

        # Task parameters
        self.radius = rospy.get_param("~radius", 0.8)                 # m
        self.forward_speed = rospy.get_param("~forward_speed", 0.3)    # m/s
        self.arrival_thres = rospy.get_param("~arrival_thres", 0.15)   # m
        self.yaw_thres = math.radians(rospy.get_param("~yaw_thres_deg", 5.0))
        self.circle_direction = rospy.get_param("~circle_direction", "ccw").lower()
        self.circle_sign = -1.0 if self.circle_direction == "cw" else 1.0

        self.rate = rospy.Rate(rospy.get_param("~rate", 20.0))

        self.current_pose = PoseStamped()
        self.pose_ready = False
        self.uav_state = UAVState()
        self.uav_control_state = Int32()
        self.stop_flag = False
        self.cmd_id = 0

        rospy.Subscriber(self.topic_prefix + "/sunray/uav_state",
                         UAVState, self.uav_state_cb, queue_size=1)
        rospy.Subscriber(self.topic_prefix + "/sunray/control_state",
                         Int32, self.uav_control_state_cb, queue_size=1)
        rospy.Subscriber(self.topic_prefix + "/sunray/stop_tutorial",
                         Empty, self.stop_cb, queue_size=1)
        rospy.Subscriber(self.topic_prefix + "/mavros/local_position/pose",
                         PoseStamped, self.pose_cb, queue_size=10)

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

        # Fallback values match the C++ examples and common Sunray message order.
        self.CMD_TAKEOFF = msg_const(UAVControlCMD, "Takeoff", 1)
        self.CMD_HOVER = msg_const(UAVControlCMD, "Hover", 2)
        self.CMD_LAND = msg_const(UAVControlCMD, "Land", 3)
        self.CMD_XYZ_POS = msg_const(UAVControlCMD, "XYZ_POS", 4)
        self.CMD_XY_VEL_Z_POS = msg_const(UAVControlCMD, "XY_VEL_Z_POS", 6)

        self.SETUP_ARM = msg_const(UAVSetup, "ARM", 1)
        self.SETUP_SET_CONTROL_MODE = msg_const(UAVSetup, "SET_CONTROL_MODE", 4)

        rospy.loginfo("circle_height_hold started.")
        rospy.loginfo("uav_name = %s", self.uav_name)
        rospy.loginfo("control topic = %s", self.topic_prefix + "/sunray/uav_control_cmd")
        rospy.loginfo("setup topic   = %s", self.topic_prefix + "/sunray/setup")
        rospy.loginfo("Use XY_VEL_Z_POS during circle to hold altitude.")

    def uav_state_cb(self, msg):
        self.uav_state = msg

    def uav_control_state_cb(self, msg):
        self.uav_control_state = msg

    def stop_cb(self, msg):
        self.stop_flag = True

    def pose_cb(self, msg):
        self.current_pose = msg
        self.pose_ready = True

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

    def takeoff(self, duration=10.0):
        rospy.loginfo("takeoff...")
        cmd = self.make_cmd()
        cmd.cmd = self.CMD_TAKEOFF
        self.publish_cmd_for(cmd, duration)

    def hover(self, duration=1.0):
        rospy.loginfo("hover %.1f s...", duration)
        cmd = self.make_cmd()
        cmd.cmd = self.CMD_HOVER
        self.publish_cmd_for(cmd, duration)

    def land(self):
        rospy.loginfo("land...")
        cmd = self.make_cmd()
        cmd.cmd = self.CMD_LAND
        self.publish_cmd_for(cmd, 0.5)

    def wait_for_pose_ready(self, timeout=5.0):
        rospy.loginfo("wait for pose: %s", self.topic_prefix + "/mavros/local_position/pose")
        t0 = rospy.Time.now()
        while not rospy.is_shutdown():
            if self.pose_ready:
                return True
            if (rospy.Time.now() - t0).to_sec() > timeout:
                rospy.logerr("No pose received. Please check topic name and MAVROS.")
                return False
            self.rate.sleep()

    def goto_point(self, x, y, z, yaw, timeout=20.0, hold_time=0.5):
        rospy.loginfo("goto point: %.2f, %.2f, %.2f, yaw %.1f deg",
                      x, y, z, math.degrees(yaw))
        t0 = rospy.Time.now()

        while not rospy.is_shutdown():
            if self.stop_flag:
                self.land()
                return False

            cmd = self.make_cmd()
            cmd.cmd = self.CMD_XYZ_POS
            cmd.desired_pos = [x, y, z]
            cmd.desired_yaw = yaw
            cmd.enable_yawRate = False
            self.publish_cmd_once(cmd)

            px, py, pz, _ = self.get_xyz_yaw()
            err = math.sqrt((px - x) ** 2 + (py - y) ** 2 + (pz - z) ** 2)

            if err < self.arrival_thres:
                rospy.loginfo("arrived, err = %.3f m", err)
                rospy.sleep(hold_time)
                return True

            if (rospy.Time.now() - t0).to_sec() > timeout:
                rospy.logwarn("goto timeout, err = %.3f m", err)
                return False

            self.rate.sleep()

    def rotate_to_yaw(self, x, y, z, yaw_target, timeout=10.0):
        rospy.loginfo("rotate to yaw %.1f deg", math.degrees(yaw_target))
        t0 = rospy.Time.now()

        while not rospy.is_shutdown():
            if self.stop_flag:
                self.land()
                return False

            cmd = self.make_cmd()
            cmd.cmd = self.CMD_XYZ_POS
            cmd.desired_pos = [x, y, z]
            cmd.desired_yaw = yaw_target
            cmd.enable_yawRate = False
            self.publish_cmd_once(cmd)

            _, _, _, yaw = self.get_xyz_yaw()
            yaw_err = self.normalize_angle(yaw_target - yaw)

            if abs(yaw_err) < self.yaw_thres:
                rospy.loginfo("yaw aligned, err = %.2f deg", math.degrees(yaw_err))
                rospy.sleep(0.5)
                return True

            if (rospy.Time.now() - t0).to_sec() > timeout:
                rospy.logwarn("rotate timeout, yaw_err = %.2f deg", math.degrees(yaw_err))
                return False

            self.rate.sleep()

    def fly_circle_height_hold(self, z_hold):
        """Fly one circle with inertial XY velocity and fixed Z position.

        This replaces XYZ_VEL_BODY. The horizontal velocity is recomputed from
        current yaw, so it is still equivalent to commanding body-forward motion,
        while the z channel is closed by the z-position controller.
        """
        yaw_rate = self.circle_sign * self.forward_speed / max(self.radius, 1e-6)
        circle_time = 2.0 * math.pi * self.radius / max(self.forward_speed, 1e-6)

        rospy.loginfo("circle height hold: z_hold=%.2f m", z_hold)
        rospy.loginfo("vx_body=%.2f m/s, yaw_rate=%.3f rad/s, time=%.2f s",
                      self.forward_speed, yaw_rate, circle_time)

        t0 = rospy.Time.now()
        while not rospy.is_shutdown():
            if self.stop_flag:
                self.land()
                return False

            elapsed = (rospy.Time.now() - t0).to_sec()
            if elapsed >= circle_time:
                rospy.loginfo("circle finished")
                return True

            _, _, _, yaw = self.get_xyz_yaw()

            # Convert desired body-forward velocity to ENU XY velocity.
            vx_enu = self.forward_speed * math.cos(yaw)
            vy_enu = self.forward_speed * math.sin(yaw)

            cmd = self.make_cmd()
            cmd.cmd = self.CMD_XY_VEL_Z_POS
            cmd.desired_pos = [0.0, 0.0, z_hold]
            cmd.desired_vel = [vx_enu, vy_enu, 0.0]
            cmd.desired_yaw_rate = yaw_rate
            cmd.enable_yawRate = True
            self.publish_cmd_once(cmd)

            self.rate.sleep()

    def run(self):
        # Avoid losing the first setup messages.
        rospy.sleep(1.0)

        # Follow the original block_pos.cpp style: arm first, then command.
        self.arm_and_enter_cmd_control()
        self.takeoff(duration=10.0)
        self.hover(duration=1.0)

        if not self.wait_for_pose_ready(timeout=5.0):
            rospy.logerr("Pose is required for circle-center calculation. Stop before circle.")
            return

        # Record the actual hover point as the circle center and height reference.
        cx, cy, cz, yaw0 = self.get_xyz_yaw()
        z_hold = cz

        rospy.loginfo("circle center = %.2f, %.2f, %.2f; yaw0 = %.1f deg",
                      cx, cy, cz, math.degrees(yaw0))

        # Fly 1 m along current body heading to the circle start point.
        start_x = cx + self.radius * math.cos(yaw0)
        start_y = cy + self.radius * math.sin(yaw0)
        start_z = z_hold

        if not self.goto_point(start_x, start_y, start_z, yaw0):
            return

        # Tangent yaw. CCW: yaw0 + 90 deg. CW: yaw0 - 90 deg.
        tangent_yaw = self.normalize_angle(yaw0 + self.circle_sign * math.pi / 2.0)

        if not self.rotate_to_yaw(start_x, start_y, start_z, tangent_yaw):
            return

        # Fly one circle while holding the hover height z_hold.
        self.fly_circle_height_hold(z_hold)

        self.hover(duration=1.0)

        # Return to the recorded takeoff/hover point and land.
        self.goto_point(cx, cy, z_hold, yaw0, timeout=20.0, hold_time=1.0)
        self.land()


if __name__ == "__main__":
    try:
        CircleHeightHoldNode().run()
    except rospy.ROSInterruptException:
        pass
