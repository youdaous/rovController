#!/usr/bin/env python
#coding=utf-8

import rospy
import numpy as np
from uuv_control_interfaces import DPControllerBase
from tf_quaternion.transformations import euler_from_quaternion

class TutorialDPController(DPControllerBase):
  def __init__(self):
      super(TutorialDPController, self).__init__(self)

      self._Kp = np.zeros(shape=(2, 2))
      self._Kd = np.zeros(shape=(2, 2))
      self._Ki = np.zeros(shape=(2, 2))

      self._int = np.zeros(shape=(2,))
      self._error_pose = np.zeros(shape=(2,))

    
      self.error_pose_last_k = np.zeros(shape=(2,))
      self.error_vel_last_k =  np.zeros(shape=(2,))
      self.tau_last_k = np.zeros(shape=(2,))

      self.tau = np.zeros(shape=(6,))

    	# PID学习参数设置
      self.learn_rate = np.array([0.4, 0.35, 0.4])

		# Do the same for the other two matrices
      if rospy.get_param('~Kp'):
          diag = rospy.get_param('~Kp')
          if len(diag) == 2:
              self._Kp = np.diag(diag)
              print 'Kp=\n', self._Kp
          else:
              # If the vector provided has the wrong dimension, raise an exception
              raise rospy.ROSException('For the Kp diagonal matrix, 6 coefficients are needed')

      if rospy.get_param('~Kd'):
          diag = rospy.get_param('~Kd')
          if len(diag) == 2:
              self._Kd = np.diag(diag)
              print 'Kd=\n', self._Kd
          else:
              # If the vector provided has the wrong dimension, raise an exception
              raise rospy.ROSException('For the Kd diagonal matrix, 6 coefficients are needed')

      if rospy.get_param('~Ki'):
          diag = rospy.get_param('~Ki')
          if len(diag) == 2:
              self._Ki = np.diag(diag)
              print 'Ki=\n', self._Ki
          else:
              # If the vector provided has the wrong dimension, raise an exception
              raise rospy.ROSException('For the Ki diagonal matrix, 6 coefficients are needed')
          
      self.w_k = np.array([self._Kp, self._Ki, self._Kd])
      self._is_init = True

  def _reset_controller(self):
      super(TutorialDPController, self)._reset_controller()
      self._error_pose = np.zeros(shape=(2,))
      self._int = np.zeros(shape=(2,))

  def update_controller(self):
      if not self._is_init:
          return False

      if not self.odom_is_init:
          return
        #   self._int = self._int + 0.5 * (self.error_pose_euler[[2, 3]] + self._error_pose) * self._dt
        #   self._error_pose = self.error_pose_euler[[2, 3]]
        #   tau_k = np.dot(self._Kp, self.error_pose_euler[[2, 3]]) + np.dot(self._Kd, self._errors['vel'][[2, 3]]) + np.dot(self._Ki, self._int)
            
        #    增量式PID
        #   error_pose_k = self.error_pose_euler[[2, 3]]
        #   error_vel_k = self._errors['vel'][[2, 3]]
        #   error_ki = 0.5 * ( error_pose_k + self.error_pose_last_k) * self._dt
        #   x_cell_input = np.array([ error_pose_k - self.error_pose_last_k, error_ki, error_vel_k - self.error_vel_last_k])
        #   tau_k = self.tau_last_k + (np.dot(self.w_k[0], x_cell_input[0]) + np.dot(self.w_k[1], x_cell_input[1]) +  np.dot(self.w_k[2], x_cell_input[2])) * self._control_saturation 
      
        #   位置式PID
      error_pose_k = self.error_pose_euler[[2, 3]]
      error_vel_k = self._errors['vel'][[2, 3]]
      self._int = self._int +  0.5 * ( error_pose_k + self.error_pose_last_k) * self._dt
      x_cell_input = np.array([error_pose_k, self._int, error_vel_k])
      tau_k = (np.dot(self.w_k[0], x_cell_input[0]) + np.dot(self.w_k[1], x_cell_input[1]) +  np.dot(self.w_k[2], x_cell_input[2])) * self._control_saturation 

      self.error_pose_last_k = error_pose_k
      self.error_vel_last_k = error_vel_k
      self.tau_last_k = tau_k
      
      if  (x_cell_input[0] > 0.001).any():
         self.w_k = self.learn_rule(self.w_k, x_cell_input[0], x_cell_input, tau_k)
        #  print('w_k:{}'.format(self.w_k))
        #   print(tau_k.shape)
        #   print(tau_k)
         
      print(self.reference_pose_inEuler())
      self.tau[2] = tau_k[0]
      self.tau[3] = tau_k[1]
      self.publish_control_wrench(self.tau)

  def learn_rule(self, w_k, error, x, y):
        """
        单神经元PID学习规则。
        w_k为神经元连接权重, 即PID系数;
        error为期望输出和实际输出的差, 即error=x[0];
        x为神经元输入, 即PID的三项输入;
        y为神经元输出, 即PID的输出。
        """
        # if error < 0.05:
        #     return w_k
        w_k_0 = np.diagonal(w_k[0]) + self.learn_rate[0] * error * y * x[0] * self._control_saturation
        w_k_1 = np.diagonal(w_k[1]) + self.learn_rate[1] * error * y * x[1] * self._control_saturation
        w_k_2 = np.diagonal(w_k[2]) + self.learn_rate[2] * error * y * x[2] * self._control_saturation
        w_k[0] = np.diag(2 / (1 + np.exp(-np.square(w_k_0))) - 1)
        w_k[0] = np.diag(2 / (1 + np.exp(-np.square(w_k_1))) - 1)
        w_k[0] = np.diag(2 / (1 + np.exp(-np.square(w_k_2))) - 1)
        # w_k = 1 / (1 + np.exp(-np.square(w_k))) 
        # w_k = w_k / sum(abs(w_k))
        # print(error)
        return w_k
  
  def reference_pose_inEuler(self):
        roll, pitch, yaw = euler_from_quaternion(self._reference['rot'])
        return np.array([self._reference['pos'], [roll, pitch, yaw]]).reshape(6, 1)
    

if __name__ == '__main__':
  print('myROVcontroller')
  rospy.init_node('myROVcontroller_node')

  try:
      node = TutorialDPController()
      rospy.spin()
  except rospy.ROSInterruptException:
      print('caught exception')
  print('exiting')