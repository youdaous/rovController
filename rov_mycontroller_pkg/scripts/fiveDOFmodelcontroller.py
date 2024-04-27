#!/usr/bin/env python
#coding=utf-8
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
        self.MassMatric = np.diag(self._vehicle_model._Mtotal[[0, 1, 2, 3, 5], [0, 1, 2, 3, 5]])
        self.MassMatric_inv = np.linalg.inv(self.MassMatric)
        self.D = -1 * np.diag(self._vehicle_model._linear_damping[[0, 1, 2, 3, 5], [0, 1, 2, 3, 5]])
        self.restoring_f = np.zeros((5,1))
        print("Mass:\n{}\nD:\n{}\nrestoring force:\n{}".format(self.MassMatric, self.D, self.restoring_f))

        self._is_init = True
        # print('frameID is {}!!!'.format(self._vehicle_model._body_frame_id))

        self.ref = np.zeros((5 , 1))
        self.tau = np.zeros(shape=(6,))

        if rospy.get_param('~ref'):
            ref = rospy.get_param('~ref')
            if len(ref) == 5:
                self.ref = np.array(ref).reshape(5, 1)
                print 'ref=\n', self.ref

        # MPC超参数
        self.N = 10 #predict steps
        self.dt = 0.1
        self.Q = np.diag([1, 1, 1,  1, 1 ]) * 1e5
        self.R = np.diag([1, 1, 1, 1, 1]) * 1e-1

        #   初始化优化器
        self.opti = casadi.Opti()
        self.U = self.opti.variable(5, self.N) #   控制序列
        self.X_vel = self.opti.variable(5, self.N+1) #   状态序列
        self.X_pose = self.opti.variable(5, self.N+1)
        self.P = self.opti.parameter(5, 1)
        self.X0_pose = self.opti.parameter(5, 1)
        self.X0_vel = self.opti.parameter(5, 1)
        self.g0 = self.opti.parameter(5, 1)

        # 初始化代价
        J = 0
        # print 'dt=\n', self._dt

        for i in range(self.N):
            self.opti.subject_to(self.X_vel[:, i+1] == self.X_vel[:, i] + self.dt * casadi.mtimes(self.MassMatric_inv, self.U[:, i] - casadi.mtimes(self.D, self.X_vel[:, i]) - self.g0))
            phi = self.X_pose[3, i]
            psi = self.X_pose[4, i]
            T_linear = np.array([[casadi.cos(psi),  - casadi.cos(phi) * casadi.sin(psi), casadi.sin(phi) * casadi.sin(psi)],
                    [casadi.sin(psi), casadi.cos(phi) * casadi.cos(psi), - casadi.sin(phi) * casadi.cos(psi)],
                    [0, casadi.sin(phi), casadi.cos(phi)]])
            T_angle = np.array([[1, 0],
                       [0, casadi.cos(phi)]])
            T_total = np.block([[T_linear, np.zeros((3,2))], [np.zeros((2,3)), T_angle]])

            for k in range(5):
                self.opti.subject_to(self.X_pose[k, i+1] == self.X_pose[k, i] + self.dt * (T_total[k, 0] * self.X_vel[0, i] + T_total[k, 1] * self.X_vel[1, i] + T_total[k, 2] * self.X_vel[2, i] + \
                                                                                           T_total[k, 3] * self.X_vel[3, i] + T_total[k, 4] * self.X_vel[4, i]))
                

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
        
        self._vehicle_model._update_restoring(use_sname=True)
        print("restoring force:{}", self._vehicle_model._g)
        self.restoring_f[2] = self._vehicle_model._g[2]
        # opti.set_initial(U[:, 0], [0, 0, 0, 0, 0, 0])
        # self.opti.set_value(self.P, self.ref)
        self.opti.set_value(self.g0, self.restoring_f)
        self.opti.set_value(self.P, self.reference_pose_inEuler())
        self.opti.set_value(self.X0_pose, self.vehicle_pose_inEuler())
        self.opti.set_value(self.X0_vel, self.vehicle_vel())
        sol = self.opti.solve()
        outforce = sol.value(self.U[:, 0])
        self.tau[0] = outforce[0]
        self.tau[1] = outforce[1]
        self.tau[2] = outforce[2]
        self.tau[3] = outforce[3]
        self.tau[5] = outforce[4]

        self.publish_control_wrench(self.tau)

    def reference_pose_inEuler(self):
        roll, pitch, yaw = euler_from_quaternion(self._reference['rot'])
        reference_rot = [roll, yaw]
        roll_vehicle, pitch_vehicle, yaw_vehicle = euler_from_quaternion(self._vehicle_model.quat)
        vehicle_rot = [roll_vehicle, yaw_vehicle]

        for i in range(len(reference_rot)):
            if abs(reference_rot[i] - vehicle_rot[i]) > np.pi:
                if reference_rot[i] > 0:
                    reference_rot[i] = -np.pi - reference_rot[i]
                else:
                    reference_rot[i] = 2 * np.pi + reference_rot[i]
        return np.array([self._reference['pos'][0], self._reference['pos'][1], self._reference['pos'][2], reference_rot[0], reference_rot[1]]).reshape(5, 1)
    
    def vehicle_pose_inEuler(self):
        roll, pitch, yaw = euler_from_quaternion(self._vehicle_model.quat)
        return np.array([self._vehicle_model.pos[0], self._vehicle_model.pos[1], self._vehicle_model.pos[2],  roll, yaw]).reshape(5, 1)
    
    def vehicle_vel(self):
        return np.array([self._vehicle_model.vel[0], self._vehicle_model.vel[1], self._vehicle_model.vel[2], self._vehicle_model.vel[3], self._vehicle_model.vel[5]]).reshape(5, 1)
    

if __name__ == '__main__':
  
  print('myROVcontroller')
  rospy.init_node('myROVcontroller_node')

  try:
      node = MPCcontroller()
      rospy.spin()
  except rospy.ROSInterruptException:
      print('caught exception')
  print('exiting')
