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
        self.MassMatric_inv = np.linalg.inv(self._vehicle_model._Mtotal)
        self.buoyance_gravity = self._vehicle_model._g
        self.D = -1 * self._vehicle_model._linear_damping
        # print("Mass:\n{}\nD:\n{}\n".format(self._vehicle_model._Mtotal, self.D))

        self._is_init = True
        # print('frameID is {}!!!'.format(self._vehicle_model._body_frame_id))

        self.ref = np.zeros((6 , 1))
        self.tau = np.zeros(shape=(6,))

        if rospy.get_param('~ref'):
            ref = rospy.get_param('~ref')
            if len(ref) == 6:
                self.ref = np.array(ref).reshape(6, 1)
                print 'ref=\n', self.ref

        # MPC超参数
        self.N = 20 #predict steps
        self.dt = 0.05
        self.Q = np.diag([1, 1, 1,  1, 1, 1]) * 1e5
        self.R = np.diag([1, 1, 1, 1, 1, 1]) * 1e-1

        #   初始化优化器
        self.opti = casadi.Opti()
        self.U = self.opti.variable(6, self.N) #   控制序列
        self.X_vel = self.opti.variable(6, self.N+1) #   状态序列
        self.X_pose = self.opti.variable(6, self.N+1)
        self.P = self.opti.parameter(6, 1)
        self.X0_pose = self.opti.parameter(6, 1)
        self.X0_vel = self.opti.parameter(6, 1)

        # 初始化代价
        J = 0
        # print 'dt=\n', self._dt

        for i in range(self.N):
            self.opti.subject_to(self.X_vel[:, i+1] == self.X_vel[:, i] + self.dt * casadi.mtimes(self.MassMatric_inv, self.U[:, i] - casadi.mtimes(self.D, self.X_vel[:, i])))
            phi = self.X_pose[3, i]
            theta = self.X_pose[4, i]
            psi = self.X_pose[5, i]
            T_linear = np.array([[casadi.cos(theta) * casadi.cos(psi), casadi.sin(phi) * casadi.sin(theta) * casadi.cos(psi) - casadi.cos(phi) * casadi.sin(psi), casadi.cos(phi) * casadi.sin(theta) * casadi.cos(psi) + casadi.sin(phi) * casadi.sin(psi)],
                    [casadi.cos(theta) * casadi.sin(psi), casadi.sin(phi) * casadi.sin(theta) * casadi.sin(psi) + casadi.cos(phi) * casadi.cos(psi), casadi.cos(phi) * casadi.sin(theta) * casadi.sin(psi) - casadi.sin(phi) * casadi.cos(psi)],
                    [-casadi.sin(theta), casadi.sin(phi) * casadi.cos(theta), casadi.cos(phi) * casadi.cos(theta)]])
            T_angle = np.array([[1, casadi.sin(phi) * casadi.tan(theta), casadi.cos(phi) * casadi.tan(theta)],
                       [0, casadi.cos(phi), -casadi.sin(phi)],
                       [0, casadi.sin(phi) / casadi.cos(theta), casadi.cos(phi) / casadi.cos(theta)]])
            T_total = np.block([[T_linear, np.zeros((3,3))], [np.zeros((3,3)), T_angle]])

            for k in range(6):
                self.opti.subject_to(self.X_pose[k, i+1] == self.X_pose[k, i] + self.dt * (T_total[k, 0] * self.X_vel[0, i] + T_total[k, 1] * self.X_vel[1, i] + T_total[k, 2] * self.X_vel[2, i] + \
                                                                                           T_total[k, 3] * self.X_vel[3, i] + T_total[k, 4] * self.X_vel[4, i] + T_total[k, 5] * self.X_vel[5, i]))
                
            # self.opti.subject_to(self.X_pose[:, i+1] == self.X_pose[:, i] + self.dt *self.X[:, i] )

            # self.opti.subject_to(self.X_pose[0, i+1] == self.X_pose[0, i] + self.dt * (T_total[0, 0] * self.X[0, i] + T_total[0, 1] * self.X[1, i] + T_total[0, 2] * self.X[2, i]))
            # self.opti.subject_to(self.X_pose[1, i+1] == self.X_pose[1, i] + self.dt * (T_total[1, 0] * self.X[0, i] + T_total[1, 1] * self.X[1, i] + T_total[1, 2] * self.X[2, i]))
            # self.opti.subject_to(self.X_pose[2, i+1] == self.X_pose[2, i] + self.dt * (T_total[2, 0] * self.X[0, i] + T_total[2, 1] * self.X[1, i] + T_total[2, 2] * self.X[2, i]))
            # self.opti.subject_to(self.X_pose[3, i+1] == self.X_pose[3, i] + self.dt * (T_total[3, 3] * self.X[3, i] + T_total[3, 4] * self.X[4, i] + T_total[3, 5] * self.X[5, i]))
            # self.opti.subject_to(self.X_pose[4, i+1] == self.X_pose[4, i] + self.dt * (T_total[4, 3] * self.X[3, i] + T_total[4, 4] * self.X[4, i] + T_total[4, 5] * self.X[5, i]))
            # self.opti.subject_to(self.X_pose[5, i+1] == self.X_pose[5, i] + self.dt * (T_total[5, 3] * self.X[3, i] + T_total[5, 4] * self.X[4, i] + T_total[5, 5] * self.X[5, i]))
            # self.opti.subject_to(self.X_pose[:, i+1] == self.X_pose[:, i] + self.dt * casadi.mtimes())
            J += casadi.mtimes([(self.X_pose[:, i+1] - self.P).T, self.Q, (self.X_pose[:, i+1] - self.P)]) + casadi.mtimes([self.U[:, i].T, self.R, self.U[:, i]])
            
        self.opti.subject_to(self.X_pose[:, 0] == self.X0_pose)
        self.opti.subject_to(self.X_vel[:, 0] == self.X0_vel)
        # self.opti.subject_to(self.U[2, :] == np.zeros((1, self.N)))
        # self.opti.subject_to(self.U[3, :] == np.zeros((1, self.N)))
        # self.opti.subject_to(self.U[4, :] == np.zeros((1, self.N)))
        # self.opti.subject_to(self.U[5, :] == np.zeros((1, self.N)))
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

        # opti.set_initial(U[:, 0], [0, 0, 0, 0, 0, 0])
        # self.opti.set_value(self.P, self.ref)
        self.opti.set_value(self.P, self.reference_pose_inEuler())
        self.opti.set_value(self.X0_pose, self.vehicle_pose_inEuler())
        self.opti.set_value(self.X0_vel, self.vehicle_vel())
        sol = self.opti.solve()
        outforce = sol.value(self.U[:, 0])

        self.publish_control_wrench(outforce)

    def reference_pose_inEuler(self):
        roll, pitch, yaw = euler_from_quaternion(self._reference['rot'])
        return np.array([self._reference['pos'], [roll, pitch, yaw]]).reshape(6, 1)
    
    def vehicle_pose_inEuler(self):
        roll, pitch, yaw = euler_from_quaternion(self._vehicle_model.quat)
        return np.array([self._vehicle_model.pos,  [roll, pitch, yaw]]).reshape(6, 1)
    
    def vehicle_vel(self):
        return self._vehicle_model.vel.reshape(6, 1)


    # def state_tran(self, forces=None, state=None):
    #     C = np.zeros((6, 6))
    #     D = -1 * self._vehicle_model._linear_damping - state[0] * self._vehicle_model._linear_damping_forward_speed
    #     for i in range(6):
    #        D[i, i] += -1 * self._vehicle_model._quad_damping[i] * casadi.fabs(state[i])
    #     acc = casadi.mtimes(self.MassMatric_inv, forces - casadi.mtimes(C+D, state))
    #     # print("D:\n{}!!!\n Mass:\n{}!!\n g:\n{}!!".format(D, self.MassMatric_inv, self.buoyance_gravity))
    #     state_next = state + acc * self.dt
    #     return state_next
    # def state_tran(self, forces=None, state=None, use_sname=True, outState_isVel=True):
    #     """Calculate inverse dynamics to obtain the acceleration vector in X-Y plane with 3*1 vectors."""
    #     # if forces is  None:
    #     #     self._acc =  casadi.MX.zeros(6, 1)
    #     # Check if the mass and inertial parameters were set
    #     if self._vehicle_model._Mtotal.sum() == 0 or forces is  None:
    #         self._acc = casadi.MX.zeros(6, 1)
    #         print('No mass: {}!!!'.format(self._vehicle_model._Mtotal))
    #     else:
    #         self._vehicle_model._update_damping()
    #         self._vehicle_model._update_coriolis()
    #         self._vehicle_model._update_restoring(use_sname=True)
    #         # Compute the vehicle's acceleration
    #         self._acc = casadi.mtimes(np.linalg.inv(self._vehicle_model._Mtotal), forces - 
    #                            casadi.mtimes(self._vehicle_model._C, state) - 
    #                            casadi.mtimes(self._vehicle_model._D, state) - 
    #                            self._vehicle_model._g)

    #     if outState_isVel:
    #         nextstate = state + self._acc * self.dt
    #     else:
    #         state = self._vehicle_model._pose + self._vehicle_model._vel * self.dt + 0.5 * self._acc * self.dt**2
    #     return nextstate
    
if __name__ == '__main__':
  
  print('myROVcontroller')
  rospy.init_node('myROVcontroller_node')

  try:
      node = MPCcontroller()
      rospy.spin()
  except rospy.ROSInterruptException:
      print('caught exception')
  print('exiting')
