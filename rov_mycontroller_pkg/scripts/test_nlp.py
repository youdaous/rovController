#!/usr/bin/env python
#coding=utf-8
from operator import inv
from sys import path
path.append(r"/home/youda/casadi-linux-py27-fadc864")
import casadi
import rospy
import numpy as np
from tf_quaternion.transformations import euler_from_quaternion
import matplotlib.pyplot as plt
import math

class MPCcontroller(object):
    def __init__(self, M, D, saturation):
        
        # 车辆模型参数
        self.MassMatric = np.diag(M)
        self.MassMatric_inv = np.linalg.inv(self.MassMatric)
        self.D = -1 * np.diag(D)
        self._control_saturation = saturation
        # self.D = 0
        print("Mass:\n{}\nD:\n{}\n".format(self.MassMatric, self.D))

        self.state_pos = np.zeros((3, 1)).reshape(3,1)
        self.state_vel = np.zeros((3, 1)).reshape(3,1)


        self.ref = np.zeros((3 , 1))
        self.tau = np.zeros(shape=(6,))

        # MPC超参数
        self.N = 20 #predict steps
        self.dt = 0.1
        self.Q = np.diag([1, 1, 0]) * 1e5
        self.R = np.diag([1, 1, 0]) * 1e-1

        #   初始化优化器
        self.opti = casadi.Opti()
        self.U = self.opti.variable(3, self.N) #   控制序列
        self.X = self.opti.variable(3, self.N+1) #   状态序列
        self.X_pose = self.opti.variable(3, self.N+1)
        self.P = self.opti.parameter(3, 1)
        self.X0 = self.opti.parameter(3, 1)
        self.X0_pose = self.opti.parameter(3, 1)

        # 初始化代价
        J = 0
        # print 'dt=\n', self._dt

        for i in range(self.N):
            self.opti.subject_to(self.X[:, i+1] == self.X[:, i] + self.dt * casadi.mtimes(self.MassMatric_inv, self.U[:, i] - casadi.mtimes(self.D, self.X[:, i])))
            psi = self.X_pose[2, i]
            T = np.array([[casadi.cos(psi), -casadi.sin(psi), 0],
                          [casadi.sin(psi), casadi.cos(psi), 0],
                          [0, 0, 1]])

            for k in range(3):
                self.opti.subject_to(self.X_pose[k, i+1] == self.X_pose[k, i] + self.dt * (T[k, 0] * self.X[0, i] + T[k, 1] * self.X[1, i] + T[k, 2] * self.X[2, i]))

            J += casadi.mtimes([(self.X_pose[:, i+1] - self.P).T, self.Q, (self.X_pose[:, i+1] - self.P)]) + casadi.mtimes([self.U[:, i].T, self.R, self.U[:, i]])
            
        self.opti.subject_to(self.X_pose[:, 0] == self.X0_pose)
        self.opti.subject_to(self.X[:, 0] == self.X0)
        self.opti.subject_to(self.U[2, :] == np.zeros((1, self.N)))
        self.opti.subject_to(self.opti.bounded(-self._control_saturation, casadi.vec(self.U), self._control_saturation))
        self.opti.minimize(J)

        opts={}
        opts["print_time"] = False
        s_opts = {}
        s_opts["print_level"] = 0
        self.opti.solver("ipopt", opts, s_opts)


    def update(self, ref, x0,x0_pose):
        # opti.set_initial(U[:, 0], [0, 0, 0, 0, 0, 0])
        self.opti.set_value(self.P, ref)
        self.opti.set_value(self.X0, x0)
        self.opti.set_value(self.X0_pose, x0_pose)

        sol = self.opti.solve()
        outforce = sol.value(self.U[:, 0])
        return outforce
    
    def simModel(self, force):
        force = force.reshape(3, 1)
        T = np.array([[math.cos(self.state_pos[2]), -math.sin(self.state_pos[2]), 0],
                     [math.sin(self.state_pos[2]), math.cos(self.state_pos[2]), 0],
                     [0, 0, 1]])
        self.state_pos = self.state_pos + self.dt * np.matmul(T, self.state_vel)
        self.state_vel = self.state_vel + self.dt * np.matmul(self.MassMatric_inv, force - np.matmul(self.D, self.state_vel))
        




    
    
if __name__ == '__main__':
    M = [11.7182, 10, 0.9157]
    D = [-11.7391, -20, -5]
    saturation = 5000
    mpc = MPCcontroller(M, D, saturation)

    ref_st = np.array([2, 1, 0]).reshape(3,1)
    x_record = []
    y_record = []
    for i in range(100):
        x0_pose = mpc.state_pos
        x0 = mpc.state_vel
        x_record.append(x0_pose[0])
        y_record.append(x0_pose[1])
        # x_record.append(x0[0])
        # print 'x0:\n', mpc.state_vel
        ref = np.sin(i * 4 * np.pi / 100) * np.ones((3, 1)) + ref_st
        force = mpc.update(ref, x0, x0_pose)
        # print 'force:\n', i
        mpc.simModel(force)
t = np.linspace(0, 10, 100)
plt.plot( x_record, y_record)
plt.plot(t, x_record)
plt.plot(t, y_record)
plt.show()
print(np.pi)