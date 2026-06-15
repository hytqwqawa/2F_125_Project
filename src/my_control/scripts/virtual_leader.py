#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import threading
import sys
from geometry_msgs.msg import PoseStamped

class VirtualLeaders:
    def __init__(self):
        rospy.init_node('virtual_leaders', anonymous=True)
        self.num_cars = rospy.get_param('~num_cars', 4)
        self.init_x = rospy.get_param('~initial_x', [0.0, -1.0, 1.0])
        self.init_y = rospy.get_param('~initial_y', [1.0, -1.0, -1.0])

        raw_final_x = rospy.get_param('~final_x', [[2.0, 1.0, 3.0]])
        raw_final_y = rospy.get_param('~final_y', [[3.0, 1.0, 1.0]])
        self.stages = self._normalize_targets(raw_final_x, raw_final_y)

        self.current_x = list(self.init_x)
        self.current_y = list(self.init_y)
        self.stage_index = 0
        self.waiting_for_command = False
        self.stage_active = False
        self.final_stage_exit_timer_set = False

        self.pubs = []
        for i in range(3):
            self.pubs.append(rospy.Publisher('/leader{}/pose'.format(i+1), PoseStamped, queue_size=10))

        rospy.Timer(rospy.Duration(0.05), self.publish_pose)
        rospy.Timer(rospy.Duration(0.5), self.check_formation_status)

        self.keyboard_thread = threading.Thread(target=self.keyboard_loop, daemon=True)
        self.keyboard_thread.start()

        rospy.loginfo("Virtual Leaders started. Press 's'+Enter to advance to the next transform after all cars reach the current target.")

    def _normalize_targets(self, final_x, final_y):
        def is_nested(seq):
            return any(isinstance(item, (list, tuple)) for item in seq)

        if not is_nested(final_x):
            final_x = [final_x]
        if not is_nested(final_y):
            final_y = [final_y]

        if len(final_x) != len(final_y):
            rospy.logerr('final_x and final_y must have the same number of stages.')
            rospy.signal_shutdown('Invalid stage parameters')
            return []

        stages = []
        for x_stage, y_stage in zip(final_x, final_y):
            if len(x_stage) != 3 or len(y_stage) != 3:
                rospy.logerr('Each stage in final_x and final_y must contain exactly 3 values.')
                rospy.signal_shutdown('Invalid stage parameters')
                return []
            stages.append((list(x_stage), list(y_stage)))
        return stages

    def publish_pose(self, event):
        for i in range(3):
            msg = PoseStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = 'map'
            msg.pose.position.x = self.current_x[i]
            msg.pose.position.y = self.current_y[i]
            self.pubs[i].publish(msg)

    def check_formation_status(self, event):
        status_dict = rospy.get_param('/formation_status', {})
        all_reached = len(status_dict) == self.num_cars and all(status_dict.values())

        if all_reached:
            if self.stage_index >= len(self.stages):
                if not self.final_stage_exit_timer_set:
                    self.final_stage_exit_timer_set = True
                    rospy.loginfo('All cars reached the final stage. Exiting in 3 seconds...')
                    rospy.Timer(rospy.Duration(3.0), self._final_stage_exit, oneshot=True)
                return

            if not self.waiting_for_command:
                rospy.loginfo('All cars reached current stage. Press s+Enter to switch to next transform, q+Enter to exit.')
                self.waiting_for_command = True
        else:
            self.waiting_for_command = False

    def keyboard_loop(self):
        while not rospy.is_shutdown():
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if not line:
                continue
            command = line.strip().lower()
            if command == 's':
                if not self.waiting_for_command:
                    print('Not ready for the next stage yet. Wait until all cars reach the current target.')
                elif self.stage_index >= len(self.stages):
                    print('No more stages available. Press q+Enter to exit.')
                else:
                    self._advance_stage()
            elif command == 'q':
                rospy.loginfo('User requested exit.')
                rospy.signal_shutdown('User requested exit.')
            elif command:
                print("Unknown command '{}' - use 's' to advance or 'q' to exit.".format(command))

    def _advance_stage(self):
        next_x, next_y = self.stages[self.stage_index]
        self.current_x = list(next_x)
        self.current_y = list(next_y)
        self.stage_index += 1
        self.waiting_for_command = False
        self.final_stage_exit_timer_set = False

        rospy.loginfo('Switching to transform stage %d.' % self.stage_index)
        current_status = rospy.get_param('/formation_status', {})
        for key in current_status:
            current_status[key] = False
        rospy.set_param('/formation_status', current_status)

    def _final_stage_exit(self, event):
        rospy.loginfo('Final stage timeout reached. Exiting now.')
        rospy.signal_shutdown('Final stage reached. Exiting.')

if __name__ == '__main__':
    try:
        VirtualLeaders()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
