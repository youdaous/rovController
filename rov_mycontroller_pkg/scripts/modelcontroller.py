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
        reference_rot = [roll, pitch, yaw]
        roll_vehicle, pitch_vehicle, yaw_vehicle = euler_from_quaternion(self._vehicle_model.quat)
        vehicle_rot = [roll_vehicle, pitch_vehicle, yaw_vehicle]

        for i in range(3):
            if abs(reference_rot[i] - vehicle_rot[i]) > np.pi:
                if reference_rot[i] > 0:
                    reference_rot[i] = -np.pi - reference_rot[i]
                else:
                    reference_rot[i] = 2 * np.pi + reference_rot[i]
        return np.array([self._reference['pos'], reference_rot]).reshape(6, 1)
    
    def vehicle_pose_inEuler(self):
        roll, pitch, yaw = euler_from_quaternion(self._vehicle_model.quat)
        return np.array([self._vehicle_model.pos,  [roll, pitch, yaw]]).reshape(6, 1)
    
    def vehicle_vel(self):
        return self._vehicle_model.vel.reshape(6, 1)
    

if __name__ == '__main__':
  
  print('myROVcontroller')
  rospy.init_node('myROVcontroller_node')

  try:
      node = MPCcontroller()
      rospy.spin()
  except rospy.ROSInterruptException:
      print('caught exception')
  print('exiting')
