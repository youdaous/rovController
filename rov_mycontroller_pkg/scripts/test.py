#!/usr/bin/env python
#coding=utf-8
from sys import path
path.append(r"/home/youda/casadi-linux-py27-fadc864")
import casadi as ca
import numpy as np

class MPC:
    def __init__(self, M, D, Q, R, N):
        self.M = np.linalg.inv(M)
        self.D = D
        self.Q = Q
        self.R = R
        self.N = N
        self.dt = 0.5
        self.opti = ca.Opti()

        # 定义优化变量
        self.U =  self.opti.variable(3, N)
        self.X = self.opti.variable(3, N+1)

        # 定义参考状态和控制输入
        self.x_ref = self.opti.parameter(3, 1)
        # self.u_ref = self.opti.parameter(B.shape[1], 1)

        # 定义初始状态
        self.x0 = self.opti.parameter(3, 1)

        # 初始化代价
        J = 0

        for i in range(N):
            # 状态更新
            self.opti.subject_to(self.X[:, i+1] == self.X[:, i] + self.dt *  ca.mtimes(self.M,  self.U[:, i] - ca.mtimes(self.D, self.X[:, i])))
            # 计算代价
            J += ca.mtimes([(self.X[:, i] - self.x_ref).T, self.Q, (self.X[:, i] - self.x_ref)]) + ca.mtimes([self.U[:, i].T, self.R, self.U[:, i]])

        # 最终状态的代价
        J += ca.mtimes([(self.X[:, -1] - self.x_ref).T, self.Q, (self.X[:, -1] - self.x_ref)])

        # 确保初始状态
        self.opti.subject_to(self.X[:, 0] == self.x0)

        # 设置目标函数
        self.opti.minimize(J)

    def solve(self, x0_val, x_ref_val, u_ref_val=None):
        self.opti.set_value(self.x0, x0_val)
        self.opti.set_value(self.x_ref, x_ref_val)
        # self.opti.set_value(self.u_ref, u_ref_val)
        self.opti.solver("ipopt")

        sol = self.opti.solve()

        return sol.value(self.U[:, 0])

# 定义系统模型和控制目标
M = np.diag([11.7182, 10, 15.468])
D = np.diag([11.7391, 20, 31.8678])
Q = np.eye(3) * 1e6
R = np.eye(3)
N = 10

# 创建 MPC 控制器
mpc = MPC(M, D, Q, R, N)

# 设置初始状态和控制目标
x0_val = np.array([0.0, 0.0, 0.0]).reshape(3,1)
x_ref_val = np.array([5, 0, 0]).reshape(3,1)
# u_ref_val = np.array([[0]])

# 解决 MPC 问题
u_opt = mpc.solve(x0_val, x_ref_val)
print("Optimal control input:", u_opt)
# print(mpc.M)
