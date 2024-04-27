#!/usr/bin/env python
#coding=utf-8
from operator import inv
from sys import path
path.append(r"/home/youda/casadi-linux-py27-fadc864")
import casadi
import rospy
import numpy as np
from uuv_control_interfaces import DPControllerBase
from tf_quaternion.transformations import euler_from_quaternion

class MPCcontroller(DPControllerBase):
    def __init__(self, is_model_based=True, list_odometry_callbacks=None, planner_full_dof=False):
        super(MPCcontroller, self).__init__(is_model_based, list_odometry_callbacks, planner_full_dof)

        # 车辆模型参数
        self.MassMatric = np.diag(self._vehicle_model._Mtotal[[0, 1, 5], [0, 1, 5]])
        self.MassMatric_inv = np.linalg.inv(self.MassMatric)
        self.D = -1 * np.diag(self._vehicle_model._linear_damping[[0, 1, 5], [0, 1, 5]])
        print("Mass:\n{}\nD:\n{}\n".format(self.MassMatric, self.D))

        if rospy.get_param('~srMass_inv'):
            diag = rospy.get_param('~srMass_inv')
            if len(diag) == 3:
                self.MassMatric_inv = np.diag(diag)
                print 'Setting SymbolicRegression MassMatric!\n'
            else:
                raise rospy.ROSException('For the Mass diagonal matrix, 3 coefficients are needed')
            
        if rospy.get_param('~srDamping'):
            diag = rospy.get_param('~~srDamping')
            if len(diag) == 3:
                self.D = -1 * np.diag(diag)
                print 'Setting SymbolicRegression Dumping\n'
            else:
                raise rospy.ROSException('For the Damping diagonal matrix, 3 coefficients are needed')

        # PID参数
        self._Kp = np.zeros(shape=(2, 2))
        self._Kd = np.zeros(shape=(2, 2))
        self._Ki = np.zeros(shape=(2, 2))
        self.K = 1000

        self._int = np.zeros(shape=(2,1))
        self._error_pose = np.zeros(shape=(2,1))


        self.error_pose_last_k = np.zeros(shape=(2,1))
        self.error_vel_last_k =  np.zeros(shape=(2,1))
        self.tau_last_k = np.zeros(shape=(2,1))

        self.tau = np.zeros(shape=(6,1))

        # PID学习参数设置
        self.learn_rate = np.array([[0.2, 0.02], [0.05, 0.05], [0.1, 0.1]])

        # Do the same for the other two matrices
        if rospy.get_param('~Kp'):
            diag = rospy.get_param('~Kp')
            if len(diag) == 2:
                self._Kp = np.array(diag).reshape(2,1)
                print 'Kp=\n', self._Kp
            else:
                # If the vector provided has the wrong dimension, raise an exception
                raise rospy.ROSException('For the Kp diagonal matrix, 6 coefficients are needed')

        if rospy.get_param('~Kd'):
            diag = rospy.get_param('~Kd')
            if len(diag) == 2:
                self._Kd = np.array(diag).reshape(2,1)
                print 'Kd=\n', self._Kd
            else:
                # If the vector provided has the wrong dimension, raise an exception
                raise rospy.ROSException('For the Kd diagonal matrix, 6 coefficients are needed')

        if rospy.get_param('~Ki'):
            diag = rospy.get_param('~Ki')
            if len(diag) == 2:
                self._Ki = np.array(diag).reshape(2,1)
                print 'Ki=\n', self._Ki
            else:
                # If the vector provided has the wrong dimension, raise an exception
                raise rospy.ROSException('For the Ki diagonal matrix, 6 coefficients are needed')
            
        self.w_k = np.array([self._Kp, self._Ki, self._Kd])

        self._is_init = True
        self.ref = np.zeros((3 , 1))
        self.tau = np.zeros(shape=(6,))

        if rospy.get_param('~ref'):
            ref = rospy.get_param('~ref')
            if len(ref) == 3:
                self.ref = np.array(ref).reshape(3, 1)
                # print 'ref=\n', self.ref

        # MPC超参数
        self.N = 20 #predict steps
        self.dt = 0.05
        self.Q = np.diag([1, 1, 1]) * 1e5
        self.R = np.diag([1, 1, 1]) * 1e-1

        #   初始化优化器
        self.opti = casadi.Opti()
        self.U = self.opti.variable(3, self.N) #   控制序列
        self.X_vel = self.opti.variable(3, self.N+1) #   状态序列
        self.X_pose = self.opti.variable(3, self.N+1)
        self.P = self.opti.parameter(3, 1)
        self.X0_pose = self.opti.parameter(3, 1)
        self.X0_vel = self.opti.parameter(3, 1)

        # 初始化代价
        J = 0
        # print 'dt=\n', self._dt

        for i in range(self.N):
            self.opti.subject_to(self.X_vel[:, i+1] == self.X_vel[:, i] + self.dt * casadi.mtimes(self.MassMatric_inv, self.U[:, i] - casadi.mtimes(self.D, self.X_vel[:, i])))
            psi = self.X_pose[2, i]
            T = np.array([[casadi.cos(psi), -casadi.sin(psi), 0],
                          [casadi.sin(psi), casadi.cos(psi), 0],
                          [0, 0, 1]])

            for k in range(3):
                self.opti.subject_to(self.X_pose[k, i+1] == self.X_pose[k, i] + self.dt * (T[k, 0] * self.X_vel[0, i] + T[k, 1] * self.X_vel[1, i] + T[k, 2] * self.X_vel[2, i]))

            J += casadi.mtimes([(self.X_pose[:, i+1] - self.P).T, self.Q, (self.X_pose[:, i+1] - self.P)]) + casadi.mtimes([self.U[:, i].T, self.R, self.U[:, i]])
            
        self.opti.subject_to(self.X_pose[:, 0] == self.X0_pose)
        self.opti.subject_to(self.X_vel[:, 0] == self.X0_vel)
        # self.opti.subject_to(self.U[2, :] == np.zeros((1, self.N)))
        self.opti.subject_to(self.opti.bounded(-self._control_saturation, casadi.vec(self.U), self._control_saturation))
        self.opti.minimize(J)

        opts={}
        opts["print_time"] = False
        s_opts = {}
        s_opts["print_level"] = 0
        self.opti.solver("ipopt", opts, s_opts)


    def update_controller(self):
        if not self._is_init:
            return False

        if not self.odom_is_init:
            return
        
        # MPC控制器更新
        # opti.set_initial(U[:, 0], [0, 0, 0, 0, 0, 0])
        self.opti.set_value(self.P, self.reference_pose_inEuler())
        self.opti.set_value(self.X0_pose, self.vehicle_pose_inEuler())
        self.opti.set_value(self.X0_vel, self.vehicle_vel())

        sol = self.opti.solve()
        outforce = sol.value(self.U[:, 0])

        self.tau[0] = outforce[0]
        self.tau[1] = outforce[1]
        self.tau[5] = outforce[2]

        # 定深PID控制器更新
        # 增量式PID
        error_pose_k = self.error_pose_euler[[2, 3]].reshape(2,1)
        error_vel_k = self._errors['vel'][[2, 3]].reshape(2,1)
        error_ki = 0.5 * ( error_pose_k + self.error_pose_last_k) * self._dt
        x_cell_input = np.array([ error_pose_k - self.error_pose_last_k, error_ki, error_vel_k - self.error_vel_last_k])
        delta_tau = ((self.w_k[0] * x_cell_input[0]) + (self.w_k[1] * x_cell_input[1]) +  (self.w_k[2] * x_cell_input[2])) * \
            self.K 
        tau_PID = self.tau_last_k + delta_tau

        # #   位置式PID
        # error_pose_k = self.error_pose_euler[[2, 3]].reshape(2,1)
        # error_vel_k = self._errors['vel'][[2, 3]].reshape(2,1)
        # self._int = self._int +  0.5 * ( error_pose_k + self.error_pose_last_k) * self._dt
        # x_cell_input = np.array([error_pose_k, self._int, error_vel_k])
        # tau_PID = (np.dot(self.w_k[0], x_cell_input[0]) + np.dot(self.w_k[1], x_cell_input[1]) +  np.dot(self.w_k[2], x_cell_input[2])) * self._control_saturation

        self.error_pose_last_k = error_pose_k
        self.error_vel_last_k = error_vel_k
        self.tau_last_k = tau_PID
      
        if  np.any(np.abs(error_pose_k) > 0.001):
            self.w_k = self.learn_rule(self.w_k, -error_pose_k, x_cell_input, delta_tau)
         
        # print(self.reference_pose_inEuler())
        self.tau[2] = tau_PID[0]
        self.tau[3] = tau_PID[1]
        # print(tau_PID)
        self.publish_control_wrench(self.tau)

    def reference_pose_inEuler(self):
        roll, pitch, yaw = euler_from_quaternion(self._reference['rot'])
        roll_vehicle, pitch_vehicle, yaw_vehicle = euler_from_quaternion(self._vehicle_model.quat)
        # if yaw < 0 and yaw_vehicle > 0:
        #     yaw = 2 * np.pi + yaw
            
        # if yaw > 0 and yaw_vehicle < 0:
        #     yaw = -np.pi - yaw

        if abs(yaw - yaw_vehicle) > np.pi:
            if yaw > 0:
                yaw = -2 * np.pi - yaw
            else:
                yaw = 2 * np.pi + yaw
        return np.array([self._reference['pos'][0], self._reference['pos'][1], yaw]).reshape(3, 1)
    
    def vehicle_pose_inEuler(self):
        roll, pitch, yaw = euler_from_quaternion(self._vehicle_model.quat)
        return np.array([self._vehicle_model.pos[0], self._vehicle_model.pos[1], yaw]).reshape(3, 1)
    
    def vehicle_vel(self):
        return np.array([self._vehicle_model.vel[0], self._vehicle_model.vel[1], self._vehicle_model.vel[5]]).reshape(3, 1)
    
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
        w_k_0 = w_k[0] + self.learn_rate[0].reshape(2, 1) * error * y * x[0]
        w_k_1 = w_k[0] + self.learn_rate[0].reshape(2, 1) * error * y * x[0]
        w_k_2 = w_k[0] + self.learn_rate[0].reshape(2, 1) * error * y * x[0]
        # print(w_k_1)
        sum = np.abs(w_k_0) + np.abs(w_k_1) + np.abs(w_k_2)
        w_k[0] = w_k_0 / sum
        w_k[1] = w_k_1 / sum
        w_k[2] = w_k_2 / sum
        # print(w_k[0])
        return w_k
    
    
if __name__ == '__main__':
  
  print('myROVcontroller')
  rospy.init_node('myROVcontroller_node')

  try:
      node = MPCcontroller()
      rospy.spin()
  except rospy.ROSInterruptException:
      print('caught exception')
  print('exiting')
