#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PN guidance with yaw control.

Task:
1) Arm and switch to CMD_CONTROL.
2) Take off and hover.
3) Fly toward a fixed target using a proportional-navigation-like
   velocity-direction law with analytic LOS-rate.
4) Keep horizontal speed magnitude fixed at 0.3 m/s.
5) Hold altitude at the hover altitude by XY_VEL_Z_POS.
6) Control yaw to follow the commanded velocity-course angle.
7) When the vehicle is close to the target, command Land.

Default target: (1.0, 0.5) in local ENU frame.

LOS-rate sign convention:
    r = p_T - p = [rx, ry]
    lambda = atan2(ry, rx)
    target is stationary, so r_dot = -v
    lambda_dot = (rx * rdot_y - ry * rdot_x) / R^2
               = (ry * vx - rx * vy) / R^2

Sign check:
    target is on +x direction: r=[R,0].
    vehicle moves along +y: v=[0,+V].
    lambda_dot = -V/R < 0.
"""

import math
import rospy
import tf.transformations

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty, Int32
from sunray_msgs.msg import UAVState, UAVControlCMD, UAVSetup


def msg_const(cls, name, fallback):
    """Use message enum constant when available; otherwise use fallback."""
    return getattr(cls, name, fallback)


class PNYawControlNode:
    def __init__(self):
        rospy.init_node("pn_yaw_control", anonymous=False)

        self.uav_id = rospy.get_param("~uav_id", 1)
        self.uav_name = rospy.get_param("~uav_name", "uav") + str(self.uav_id)
        self.topic_prefix = "/" + self.uav_name

        # ---------- Task parameters ----------
        self.target_x = rospy.get_param("~target_x", 2.0)
        self.target_y = rospy.get_param("~target_y", 0.5)

        self.speed = rospy.get_param("~speed", 0.3)                      # [m/s]
        self.N = rospy.get_param("~nav_gain", 3.0)                       # PN navigation constant
        self.max_course_rate = rospy.get_param("~max_course_rate", 0.8)  # [rad/s]
        self.land_radius = rospy.get_param("~land_radius", 0.10)         # [m]
        self.timeout = rospy.get_param("~timeout", 60.0)                 # [s]
        self.takeoff_wait = rospy.get_param("~takeoff_wait", 10.0)
        self.hover_wait = rospy.get_param("~hover_wait", 1.0)

        # Yaw control:
        #   "course": yaw tracks the commanded velocity direction gamma_cmd.
        #   "los"   : yaw points directly to the target LOS.
        # For PN trajectory experiments, "course" is usually more natural.
        self.yaw_mode = rospy.get_param("~yaw_mode", "course").lower()

        # Optional small LOS capture term. Set to 0.0 for stricter PN-only behavior.
        self.los_capture_gain = rospy.get_param("~los_capture_gain", 0.15)

        # Range lower bound to avoid singular LOS-rate near the target.
        self.R_min_for_los_rate = rospy.get_param("~R_min_for_los_rate", 0.15)

        self.rate_hz = rospy.get_param("~rate", 20.0)
        self.rate = rospy.Rate(self.rate_hz)

        # ---------- State ----------
        self.current_pose = PoseStamped()
        self.pose_ready = False
        self.uav_state = UAVState()
        self.uav_control_state = Int32()
        self.stop_flag = False
        self.cmd_id = 0

        self.gamma_cmd = None
        self.last_time = None

        # Actual horizontal velocity in ENU, updated from UAVState.velocity.
        self.vel_ready = False
        self.vx_actual = 0.0
        self.vy_actual = 0.0

        # Last commanded ENU velocity, used only as fallback if actual velocity is unavailable.
        self.vx_cmd_last = 0.0
        self.vy_cmd_last = 0.0

        # ---------- ROS I/O ----------
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

        # Fallback enum values match the existing C++ examples.
        self.CMD_TAKEOFF = msg_const(UAVControlCMD, "Takeoff", 1)
        self.CMD_HOVER = msg_const(UAVControlCMD, "Hover", 2)
        self.CMD_LAND = msg_const(UAVControlCMD, "Land", 3)
        self.CMD_XY_VEL_Z_POS = msg_const(UAVControlCMD, "XY_VEL_Z_POS", 6)

        self.SETUP_ARM = msg_const(UAVSetup, "ARM", 1)
        self.SETUP_SET_CONTROL_MODE = msg_const(UAVSetup, "SET_CONTROL_MODE", 4)

        rospy.loginfo("pn_yaw_control started.")
        rospy.loginfo("uav_name = %s", self.uav_name)
        rospy.loginfo("target = (%.2f, %.2f), speed = %.2f m/s, N = %.2f, yaw_mode = %s",
                      self.target_x, self.target_y, self.speed, self.N, self.yaw_mode)

    # ---------- Callbacks ----------
    def uav_state_cb(self, msg):
        self.uav_state = msg

        # sunray_msgs/UAVState in this project provides velocity[0:3] in local ENU.
        try:
            self.vx_actual = msg.velocity[0]
            self.vy_actual = msg.velocity[1]
            self.vel_ready = True
        except Exception:
            self.vel_ready = False

    def uav_control_state_cb(self, msg):
        self.uav_control_state = msg

    def stop_cb(self, msg):
        self.stop_flag = True

    def pose_cb(self, msg):
        self.current_pose = msg
        self.pose_ready = True

    # ---------- Math ----------
    @staticmethod
    def normalize_angle(a):
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def sat(x, xmin, xmax):
        return max(xmin, min(xmax, x))

    def get_xyz_yaw(self):
        p = self.current_pose.pose.position
        q = self.current_pose.pose.orientation
        _, _, yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        return p.x, p.y, p.z, yaw

    def compute_body_velocity(self, vx_enu, vy_enu, yaw):
        """For monitoring: convert ENU velocity to body-frame components."""
        vx_b = math.cos(yaw) * vx_enu + math.sin(yaw) * vy_enu
        vy_b = -math.sin(yaw) * vx_enu + math.cos(yaw) * vy_enu
        return vx_b, vy_b

    def analytic_los_rate(self, rx, ry, vx, vy):
        """Analytic LOS-rate for stationary target.

        r = target - vehicle = [rx, ry]
        v = vehicle inertial velocity = [vx, vy]

        lambda_dot = (ry * vx - rx * vy) / R^2
        """
        R2 = rx * rx + ry * ry
        R2 = max(R2, self.R_min_for_los_rate ** 2)
        return (ry * vx - rx * vy) / R2

    # ---------- Message helpers ----------
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

    # ---------- Basic task commands ----------
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

    # ---------- Guidance ----------
    def pn_guidance_loop(self, z_hold):
        rospy.loginfo("start PN guidance with yaw control.")
        rospy.loginfo("height hold z = %.2f m", z_hold)

        t_start = rospy.Time.now()
        self.last_time = rospy.Time.now()

        px, py, _, yaw = self.get_xyz_yaw()

        # Initial velocity-course angle. Start from current yaw direction.
        self.gamma_cmd = yaw

        self.vx_cmd_last = self.speed * math.cos(self.gamma_cmd)
        self.vy_cmd_last = self.speed * math.sin(self.gamma_cmd)

        while not rospy.is_shutdown():
            if self.stop_flag:
                self.land()
                return

            now = rospy.Time.now()
            dt = (now - self.last_time).to_sec()
            if dt <= 1e-3:
                dt = 1.0 / self.rate_hz
            self.last_time = now

            px, py, pz, yaw = self.get_xyz_yaw()
            rx = self.target_x - px
            ry = self.target_y - py
            R = math.hypot(rx, ry)

            if R < self.land_radius:
                rospy.loginfo("target reached: R = %.3f m < %.3f m", R, self.land_radius)
                self.land()
                return

            if (now - t_start).to_sec() > self.timeout:
                rospy.logwarn("PN timeout. R = %.3f m. Land for safety.", R)
                self.land()
                return

            los = math.atan2(ry, rx)

            # Use actual ENU velocity for analytic LOS-rate if available.
            # Fallback to last command during the first few samples if needed.
            if self.vel_ready:
                vx_for_los = self.vx_actual
                vy_for_los = self.vy_actual
            else:
                vx_for_los = self.vx_cmd_last
                vy_for_los = self.vy_cmd_last

            # Analytic LOS-rate with sign:
            # lambda_dot = (ry * vx - rx * vy) / R^2.
            los_dot = self.analytic_los_rate(rx, ry, vx_for_los, vy_for_los)

            # Closing speed is positive when the velocity points toward the target.
            rhat_x = rx / max(R, 1e-6)
            rhat_y = ry / max(R, 1e-6)
            v_closing = vx_for_los * rhat_x + vy_for_los * rhat_y
            v_closing = max(0.0, v_closing)

            # Proportional-navigation course-rate command:
            # gamma_dot = N * (Vc / V) * lambda_dot.
            gamma_dot_pn = self.N * (v_closing / max(self.speed, 1e-6)) * los_dot

            # A small capture term improves convergence when LOS-rate is tiny.
            # Set _los_capture_gain:=0.0 for stricter PN-only behavior.
            gamma_err = self.normalize_angle(los - self.gamma_cmd)
            gamma_dot_cap = self.los_capture_gain * gamma_err

            gamma_dot_cmd = gamma_dot_pn + gamma_dot_cap
            gamma_dot_cmd = self.sat(gamma_dot_cmd,
                                     -self.max_course_rate,
                                     self.max_course_rate)

            self.gamma_cmd = self.normalize_angle(self.gamma_cmd + gamma_dot_cmd * dt)

            # Fixed-speed horizontal velocity command in ENU.
            vx_enu = self.speed * math.cos(self.gamma_cmd)
            vy_enu = self.speed * math.sin(self.gamma_cmd)
            self.vx_cmd_last = vx_enu
            self.vy_cmd_last = vy_enu

            # Yaw command.
            if self.yaw_mode == "los":
                yaw_cmd = los
            else:
                yaw_cmd = self.gamma_cmd

            yaw_error = self.normalize_angle(yaw_cmd - yaw)

            # Body-frame forward/lateral velocity ratio for monitoring/debug.
            vx_body, vy_body = self.compute_body_velocity(vx_enu, vy_enu, yaw)

            # Use XY_VEL_Z_POS to hold altitude.
            # enable_yawRate = False means the field desired_yaw is used.
            cmd = self.make_cmd()
            cmd.cmd = self.CMD_XY_VEL_Z_POS
            cmd.desired_pos = [0.0, 0.0, z_hold]
            cmd.desired_vel = [vx_enu, vy_enu, 0.0]
            cmd.desired_yaw = yaw_cmd
            cmd.desired_yaw_rate = 0.0
            cmd.enable_yawRate = False
            self.publish_cmd_once(cmd)

            rospy.loginfo_throttle(
                1.0,
                "R=%.2f, LOS=%.1f deg, los_dot=%.3f rad/s, "
                "gamma=%.1f deg, yaw_cmd=%.1f deg, yaw_err=%.1f deg, "
                "vx_b=%.2f, vy_b=%.2f, |v|=%.2f",
                R,
                math.degrees(los),
                los_dot,
                math.degrees(self.gamma_cmd),
                math.degrees(yaw_cmd),
                math.degrees(yaw_error),
                vx_body,
                vy_body,
                math.hypot(vx_body, vy_body)
            )

            self.rate.sleep()

    def run(self):
        rospy.sleep(1.0)

        self.arm_and_enter_cmd_control()
        self.takeoff()
        self.hover(self.hover_wait)

        if not self.wait_for_pose_ready(timeout=5.0):
            rospy.logerr("Pose is required. Stop before PN guidance.")
            return

        _, _, z_hold, yaw0 = self.get_xyz_yaw()
        rospy.loginfo("hover height = %.2f m, initial yaw = %.1f deg",
                      z_hold, math.degrees(yaw0))

        self.pn_guidance_loop(z_hold)


if __name__ == "__main__":
    try:
        PNYawControlNode().run()
    except rospy.ROSInterruptException:
        pass
