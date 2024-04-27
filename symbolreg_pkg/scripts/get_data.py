#!/usr/bin/env python
#coding=utf-8

import numpy as np
import math
import rospy
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Imu
from customer_msgs.msg import Srout
from uuv_control_interfaces.vehicle import Vehicle
from uuv_control_interfaces import DPControllerBase

class getdata(DPControllerBase):
    def __init__(self, is_model_based=False, list_odometry_callbacks=None, planner_full_dof=False):
        super(getdata, self).__init__(is_model_based, list_odometry_callbacks, planner_full_dof)

        self._is_init = True
        self.obs_state = Srout()
        self.time_start = rospy.Time.now()
        self.is_imu_start = False
        self.force = Vector3()
        self.accel = Vector3()
        self.vel_last = self._vehicle_model.vel

        self.tau = np.zeros((6, 1))

        self.state_pub = rospy.Publisher('state_out', Srout, queue_size=1)
        self.imu_sub = rospy.Subscriber('imu_data', Imu, self.imu_callback, queue_size=1)

    def imu_callback(self, msg):
        if not self.is_imu_start:
            self.psi_vel_last = msg.angular_velocity.z
            self.time_last = msg.header.stamp
            self.is_imu_start = True
        else:
            self.obs_state.accel.x = msg.linear_acceleration.x
            self.obs_state.accel.y = msg.linear_acceleration.y
            self.obs_state.accel.z = (msg.angular_velocity.z - self.psi_vel_last) / (msg.header.stamp - self.time_last).to_sec()
            self.psi_vel_last = msg.angular_velocity.z
            self.time_last = msg.header.stamp

    # def step_fun(self, x, )


    def update_controller(self):
        if not self._is_init:
            return False
        if not self.odom_is_init:
            return
        timenow = rospy.Time.now()
        deltaTime = (timenow - self.time_start).to_sec()
        omega = 2 * math.pi / 10
        if deltaTime > 0 and deltaTime < 10:
            self.force.x = math.sin(omega * deltaTime) *500
            self.force.y = 0.0
            self.force.z = 0.0
        elif deltaTime >= 10 and deltaTime < 20:
            self.force.x = 0.0
            self.force.y = math.sin(omega * deltaTime) *500
            self.force.z = 0.0
        elif deltaTime >=20 and deltaTime < 30:
            self.force.x = 0.0
            self.force.y = 0.0
            self.force.z = math.sin(omega * deltaTime) *250
        else:
            self.force.x = 0.0
            self.force.y = 0.0
            self.force.z = 0.0


        self.obs_state.header.stamp = timenow
        self.obs_state.force = self.force
        self.obs_state.vel = Vector3(self._vehicle_model.vel[0], self._vehicle_model.vel[1], self._vehicle_model.vel[5])

        self.tau[0] = self.force.x
        self.tau[1] = self.force.y
        self.tau[5] = self.force.z

        self.publish_control_wrench(self.tau)
        self.state_pub.publish(self.obs_state)

        
if __name__ == '__main__':
  
  print('myROVcontroller')
  rospy.init_node('myROVcontroller_node')

  try:
      node = getdata()
      rospy.spin()
  except rospy.ROSInterruptException:
      print('caught exception')
  print('exiting')