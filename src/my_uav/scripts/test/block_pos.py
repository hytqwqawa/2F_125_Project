#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Python version of block_pos.cpp.
Function:
  1) arm the UAV;
  2) switch to CMD_CONTROL;
  3) take off;
  4) hover;
  5) fly through a square waypoint sequence using XYZ_POS;
  6) land after finishing or when /sunray/stop_tutorial is received.
"""

import math
import rospy

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty, Int32
from sunray_msgs.msg import UAVState, UAVControlCMD, UAVSetup


class BlockPosNode:
    def __init__(self):
        # Keep the node name consistent with block_pos.cpp.
        rospy.init_node("takeoff_node", anonymous=False)

        self.rate = rospy.Rate(20.0)

        self.uav_id = rospy.get_param("~uav_id", 1)
        self.uav_name_prefix = rospy.get_param("~uav_name", "uav")
        self.uav_name = f"{self.uav_name_prefix}{self.uav_id}"
        self.topic_prefix = f"/{self.uav_name}"

        self.current_pose = PoseStamped()
        self.uav_state = UAVState()
        self.uav_control_state = Int32()
        self.stop_flag = False

        rospy.Subscriber(
            self.topic_prefix + "/sunray/uav_state",
            UAVState,
            self.uav_state_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            self.topic_prefix + "/sunray/control_state",
            Int32,
            self.uav_control_state_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            self.topic_prefix + "/sunray/stop_tutorial",
            Empty,
            self.stop_tutorial_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            self.topic_prefix + "/mavros/local_position/pose",
            PoseStamped,
            self.pose_cb,
            queue_size=10,
        )

        self.control_cmd_pub = rospy.Publisher(
            self.topic_prefix + "/sunray/uav_control_cmd",
            UAVControlCMD,
            queue_size=1,
        )
        self.uav_setup_pub = rospy.Publisher(
            self.topic_prefix + "/sunray/setup",
            UAVSetup,
            queue_size=1,
        )

        self.cmd_id = 0

        # Use message constants when available. The numeric fallbacks match the
        # values used explicitly in block_pos.cpp for ARM/CMD_CONTROL/Takeoff/Hover/Land.
        self.CMD_TAKEOFF = getattr(UAVControlCMD, "Takeoff", 1)
        self.CMD_HOVER = getattr(UAVControlCMD, "Hover", 2)
        self.CMD_LAND = getattr(UAVControlCMD, "Land", 3)
        self.CMD_XYZ_POS = getattr(UAVControlCMD, "XYZ_POS", 4)

        self.SETUP_ARM = getattr(UAVSetup, "ARM", 1)
        self.SETUP_SET_CONTROL_MODE = getattr(UAVSetup, "SET_CONTROL_MODE", 4)

        rospy.loginfo(">>>>>>>>>>>>>>>> %s <<<<<<<<<<<<<<<<", rospy.get_name())
        rospy.loginfo("uav_id   : %d", self.uav_id)
        rospy.loginfo("uav_name : %s", self.uav_name)

    def uav_state_cb(self, msg):
        self.uav_state = msg

    def uav_control_state_cb(self, msg):
        self.uav_control_state = msg

    def stop_tutorial_cb(self, _msg):
        self.stop_flag = True

    def pose_cb(self, msg):
        self.current_pose = msg

    def make_default_cmd(self):
        cmd = UAVControlCMD()
        cmd.header.stamp = rospy.Time.now()
        cmd.cmd_id = self.cmd_id
        cmd.cmd = self.CMD_HOVER

        cmd.desired_pos[0] = 0.0
        cmd.desired_pos[1] = 0.0
        cmd.desired_pos[2] = 0.0
        cmd.desired_vel[0] = 0.0
        cmd.desired_vel[1] = 0.0
        cmd.desired_vel[2] = 0.0
        cmd.desired_acc[0] = 0.0
        cmd.desired_acc[1] = 0.0
        cmd.desired_acc[2] = 0.0
        cmd.desired_att[0] = 0.0
        cmd.desired_att[1] = 0.0
        cmd.desired_att[2] = 0.0
        cmd.desired_yaw = 0.0
        cmd.desired_yaw_rate = 0.0
        cmd.enable_yawRate = False
        return cmd

    def publish_cmd(self, cmd):
        cmd.header.stamp = rospy.Time.now()
        cmd.cmd_id = self.cmd_id
        self.cmd_id += 1
        self.control_cmd_pub.publish(cmd)

    def publish_setup(self, setup_cmd, control_state=""):
        setup = UAVSetup()
        setup.cmd = setup_cmd
        setup.control_state = control_state
        self.uav_setup_pub.publish(setup)

    def land(self):
        rospy.loginfo("land")
        cmd = self.make_default_cmd()
        cmd.cmd = self.CMD_LAND
        self.publish_cmd(cmd)
        rospy.sleep(0.5)

    def point_reached(self, x, y, z, threshold=0.15):
        # Same logic as block_pos.cpp: each axis error must be smaller than 0.15 m.
        px = self.current_pose.pose.position.x
        py = self.current_pose.pose.position.y
        pz = self.current_pose.pose.position.z
        return (
            abs(px - x) < threshold
            and abs(py - y) < threshold
            and abs(pz - z) < threshold
        )

    def run(self):
        rospy.sleep(0.5)

        # Arm.
        rospy.loginfo("arm")
        self.publish_setup(self.SETUP_ARM)
        rospy.sleep(1.0)

        # Switch to CMD_CONTROL.
        rospy.loginfo("switch CMD_CONTROL")
        self.publish_setup(self.SETUP_SET_CONTROL_MODE, "CMD_CONTROL")
        rospy.sleep(1.0)

        # Takeoff.
        rospy.loginfo("takeoff")
        cmd = self.make_default_cmd()
        cmd.cmd = self.CMD_TAKEOFF
        cmd.cmd_id = 0
        self.control_cmd_pub.publish(cmd)
        self.cmd_id = max(self.cmd_id, 1)
        rospy.sleep(10.0)

        # Hover.
        rospy.loginfo("hover")
        cmd = self.make_default_cmd()
        cmd.cmd = self.CMD_HOVER
        self.publish_cmd(cmd)
        rospy.sleep(2.0)

        # Same square waypoints as block_pos.cpp.
        vertices = [
            (0.9, -0.9, 0.8),
            (0.9, 0.9, 0.8),
            (-0.9, 0.9, 0.8),
            (-0.9, -0.9, 0.8),
            (0.9, -0.9, 0.8),
            (0.0, 0.0, 0.8),
        ]

        for x, y, z in vertices:
            rospy.loginfo("go to point: (%.2f %.2f %.2f)", x, y, z)

            while not rospy.is_shutdown():
                if self.stop_flag:
                    self.land()
                    return

                cmd = self.make_default_cmd()
                cmd.cmd = self.CMD_XYZ_POS
                cmd.desired_pos[0] = x
                cmd.desired_pos[1] = y
                cmd.desired_pos[2] = z
                cmd.desired_yaw = 0.0
                cmd.enable_yawRate = False
                self.publish_cmd(cmd)

                if self.point_reached(x, y, z, threshold=0.15):
                    rospy.sleep(1.0)
                    break

                self.rate.sleep()

        self.land()


if __name__ == "__main__":
    try:
        BlockPosNode().run()
    except rospy.ROSInterruptException:
        pass
