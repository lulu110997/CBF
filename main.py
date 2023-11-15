# General libraries
import sys
import matplotlib.pyplot as plt
import numpy as np
import math

# PC_RTB
import roboticstoolbox as rtb
from spatialmath import SE3, SE2

# Symbolic analysis
import sympy as sym
import IPython.display as disp

# QP solver
from qpsolvers import solve_ls


SEED = 19  # Seed for IK solver
FREQ = 0.001
TIME = 5
STEPS = int(TIME/FREQ)

def obtain_cbf():
    x_ee = sym.symbols('x_ee:2')
    x_obs = sym.symbols('x_obs:2')
    x_ee = sym.Matrix([[x_ee[0], x_ee[1]]])
    x_obs = sym.Matrix([[x_obs[0], x_obs[1]]])
    R = sym.Symbol('R')

    # This provides a much more complicated hx_dot
    alt_hx = sym.sqrt(
        (x_ee[0] - x_obs[0])**2 +
        (x_ee[1] - x_obs[1])**2
    ) - R

    hx = sym.sqrt(
        (x_ee[0] - x_obs[0])**2 +
        (x_ee[1] - x_obs[1])**2
    )**2 - R**2

    hx_dot = sym.diff(hx, x_ee)
    # print(hx_dot)
    # sym.preview(hx_dot)

    hx = sym.lambdify([x_ee, x_obs, R], expr=hx)
    hx_dot = sym.lambdify([x_ee, x_obs, R], expr=hx_dot)
    return hx, hx_dot

def abc():
    robot = rtb.DHRobot(
        [
            rtb.RevoluteDH(a=2),
            rtb.RevoluteDH(a=2)
        ], name="2D_robot", base=SE3(-0.5, 0, 0))

    t = np.arange(0, TIME, FREQ)  # time, 5/freq points
    T0 = SE3(0, 0, 0)  # initial pose
    T1 = SE3(2, 2.5, 0)  # final pose
    Ts = rtb.tools.trajectory.ctraj(T0, T1, t)

    # robot_fig = robot.plot([0, 0])
    # robot_fig.ax.set(xlim=(-1.5, 2.6), ylim=(-1.5, 2.6))

    ik_sol = robot.ikine_LM(Ts, mask=[1, 1, 0, 0, 0, 0], seed=SEED)
    # xs = [] #np.zeros((STEPS, 2))
    # for q in ik_sol.q:
    #     robot.q = q  # set the robot configuration
    #     xs.append(robot.fkine(q).t)
    #     # robot_fig.step()
    # # robot_fig.hold()

    qd = np.concatenate((np.zeros((1, 2)), np.diff(ik_sol.q, axis=0)/FREQ), axis=0)

    return robot, Ts.t[:, :2], ik_sol.q, qd


GAMMA = 1  # Used in CBF calculation
K = 2.5  # P gain for position controller
RADIUS = 0.3  # Circle radius
x_obs = np.array([1, 1])  # Obstacle position
LB = np.array([-1, -1])  # Lower bound
UB = np.array([1, 1])  # Upper bound
# x_ee = [0, 0, 0]

hx, hx_dot = obtain_cbf()
robot, xs, qs, qds = abc()

current_x = xs[0]
current_q = qs[0]

# robot_fig = robot.plot(current_q)
# robot_fig.ax.set(xlim=(-1.5, 2.6), ylim=(-1.5, 2.6))
# robot_fig.ax.view_init(elev=90, azim=90, roll=0)
#
# u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
# x = RADIUS*np.cos(u)*np.sin(v) + 1
# y = RADIUS*np.sin(u)*np.sin(v) + 1
# z = np.cos(v)
# robot_fig.ax.plot_wireframe(x, y, z, color="r")

x_online = np.zeros((STEPS, 2))
q_online = np.zeros((STEPS, 2))

for idx in range(STEPS):
    x_online[idx] = current_x.reshape((1, 2))
    q_online[idx] = current_q.reshape((1, 2))

    R = np.identity(2)  # Position we are trying to track
    s = K*(xs[idx] - current_x).reshape(2, 1)
    G = -hx_dot(xs[idx], x_obs, RADIUS)*np.identity(2)  # CBF derivative, hx_dot
    h = np.array([GAMMA * hx(xs[idx], x_obs, RADIUS), GAMMA * hx(xs[idx], x_obs, RADIUS)]).reshape(2, 1)  # CBF exponential gamma*hx
    # print(R)
    # print(s)
    # print(G)
    # print(h)

    xd_des = solve_ls(R, s, G, h, lb=LB, ub=UB, solver="osqp")  # osqp or cvxopt
    next_x = current_x + xd_des*FREQ  # xs[idx+1]

    jac = robot.jacob0(current_q)[:2, :]
    qd = np.linalg.pinv(jac) @ xd_des.reshape(2, 1)
    next_q = current_q.reshape(2, 1) + qd*FREQ
    robot.q = next_q

    current_x = next_x
    current_q = next_q

# for idx in range(0, STEPS, 10):
#     robot.q = q_online[idx]
#     robot_fig.step()
#
# robot_fig.hold()

circle1 = plt.Circle((1, 1), RADIUS, color='r')
xy_fig, xy_ax = plt.subplots()
xy_ax.add_patch(circle1)
xy_ax.plot(x_online[:, 0], x_online[:, 1])
xy_ax.plot(xs[:, 0], xs[:, 1])


# qd_fig, qd_ax = plt.subplots(nrows=3, ncols=2, constrained_layout=True)
# qd_ax = qd_ax.reshape(-1)
#
# qd_ax[0].title.set_text("Offline velocity joint 1")
# qd_ax[0].plot(qds[:, 0])
#
# qd_ax[1].title.set_text("Offline velocity joint 2")
# qd_ax[1].plot(qds[:, 1])

# qd_ax[2].title.set_text("Online velocity joint 1")
# qd_ax[2].plot(q_des_data[:, 0])
#
# qd_ax[3].title.set_text("Online velocity joint 2")
# qd_ax[3].plot(q_des_data[:, 1])
#
# qd_ax[4].title.set_text("Velocity error joint 1")
# qd_ax[4].plot(q_des_data[:, 0] - qds[:, 0])
#
# qd_ax[5].title.set_text("Velocity error joint 2")
# qd_ax[5].plot(q_des_data[:, 1] - qds[:, 1])

plt.show()
