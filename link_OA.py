# General libraries
import sys
import matplotlib.pyplot as plt
import numpy as np
import math

# PC_RTB
import roboticstoolbox as rtb
import spatialmath
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
CIRCLES_ON_LINKS = 5  # Number of circles on each link
NUM_JOINTS = 2  # 2DoF robot

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

def obtain_link_centres(q):
    link_pose = robot.fkine_all(q)  # Pose of each link in world frame
    NUM_FRAMES = link_pose.N
    num_circles = NUM_JOINTS*CIRCLES_ON_LINKS
    link_circles = np.full((num_circles-1, NUM_JOINTS), np.NAN)
    circle_idx = 0

    for idx in range(NUM_FRAMES-1):
        x_centres = np.linspace(start=link_pose[idx].t[0], stop=link_pose[idx+1].t[0], num=CIRCLES_ON_LINKS)
        y_centres = np.linspace(start=link_pose[idx].t[1], stop=link_pose[idx+1].t[1], num=CIRCLES_ON_LINKS)
        # Don't include the overlapping joints
        temp = np.vstack((x_centres, y_centres)).transpose()[1:, :]
        link_circles[circle_idx:CIRCLES_ON_LINKS*(idx+1)-1] = temp
        circle_idx += CIRCLES_ON_LINKS
    temp = np.where(~np.isnan(link_circles))
    link_circles = link_circles[temp].reshape(-1, 2)  # Remove NaN due to duplicates from overlapping joints

    return link_circles

def create_sphere(radii, centre):
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = radii * np.cos(u) * np.sin(v) + centre[0]
    y = radii * np.sin(u) * np.sin(v) + centre[1]
    z = radii * np.cos(v)
    return x, y, z

def obtain_cbf():
    x_ee = sym.symbols('x_ee:2')
    x_obs = sym.symbols('x_obs:2')
    x_ee = sym.Matrix([[x_ee[0], x_ee[1]]])
    x_obs = sym.Matrix([[x_obs[0], x_obs[1]]])
    Rw = sym.Symbol('Rw')
    R = sym.Symbol('R')

    hx = sym.sqrt(
        (x_ee[0] - x_obs[0])**2 +
        (x_ee[1] - x_obs[1])**2
    )**2 - (R + Rw)**2

    hx_dot = sym.diff(hx, x_ee)
    # print(hx_dot)
    # sym.preview(hx_dot)

    hx = sym.lambdify([x_ee, x_obs, R, Rw], expr=hx)
    hx_dot = sym.lambdify([x_ee, x_obs, R, Rw], expr=hx_dot)
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


hx, hx_dot = obtain_cbf()
robot, xs, qs, qds = abc()

# Obtain initial positions and values to save
current_x = xs[0]
current_q = qs[0]
x_tgt = np.array([2, 2.5]).reshape(current_x.shape)
x_online = np.zeros((STEPS, 2))
q_online = np.zeros((STEPS, 2))

# Initialise plot
robot_fig = robot.plot(current_q)
robot_fig.ax.set(xlim=(-1.5, 2.6), ylim=(-1.5, 2.6))
robot_fig.ax.view_init(elev=90, azim=90, roll=0)

# Plot obstacle in environment
x_obs = np.array([1, 1])
x, y, z = create_sphere(0.3, x_obs)
robot_fig.ax.plot_wireframe(x, y, z, color="r")

# Plot initial circles on link for OA vis
circle_centres = obtain_link_centres(current_q)
circles = []

# CBF parameters
# GAMMA = [0, 1, 2,... ,EE weight]
# GAMMA = [2, 2, 3, 3, 2, 1, 5, 5, 2, 2, 2]  # Used in CBF calculation
# GAMMA = [5, 5, 1, 2, 3, 3, 2, 2]
# GAMMA = [1, 2, 3, 4, 5, 6, 7, 8]
# GAMMA = [8, 7, 6, 5, 4, 3, 2, 1]
# GAMMA = [5, 5, 1, 1, 1, 1, 4, 4]  # Gamma near elbow joints too low, cannot reach end goal
# GAMMA = [5, 5, 2, 8, 8, 1, 4, 4]  # Gamma near elbow joints too high, cannot find solution that satisfies BF
# GAMMA = [5, 5, 2, 6, 6, 1, 4, 4]  # Reaches goal but EE gets too close around the obstacle + slow to reach
# GAMMA = [5, 5, 2, 6, 6, 1, 4, 2]  # Does not rach goal when using a lower gamma for EE
# GAMMA = [5, 5, 2, 6, 6, 1, 2, 2]  # Does not rach goal when using a lower gamma for EE and EE-1 sphere. Must be reacting too soon
# GAMMA = [5, 5, 2, 6, 6, 1, 1, 4]  # Reache goal faster when using a lower gamma for EE-1 sphere to react sooner w/o driving it to the wrong side
GAMMA = [5, 5, 8, 8, 8, 1, 1, 4]  # Reache goal faster when using a lower gamma for EE-1 sphere and allows spheres 3-5 to be closer to obstacle
K = 2.5  # P gain for position controller
RADIUS = 0.3  # Circle radius
Rw = 0.2  # Radius around the EE
UB = np.array([1, 1])  # Upper bound
LB = np.array([-1, -1])  # Lower bound
G = np.zeros(circle_centres.shape)
h = np.zeros((circle_centres.shape[0],))

for idx, c in enumerate(circle_centres):
    x, y, z = create_sphere(Rw, c)
    circles.append(robot_fig.ax.plot_wireframe(x, y, z, color="r"))

for idx in range(STEPS):
    x_online[idx] = current_x.reshape((1, 2))
    q_online[idx] = current_q.reshape((1, 2))

    R = np.identity(2)  # Position we are trying to track
    # s = K*(xs[idx] - current_x)
    s = K*(x_tgt - current_x)
    circle_centres = obtain_link_centres(current_q.reshape(2,))
    for jdx in range(G.shape[0]):
        G[jdx] = -hx_dot(circle_centres[jdx], x_obs, RADIUS, Rw)  # CBF derivative
        h[jdx] = GAMMA[jdx] * hx(circle_centres[jdx], x_obs, RADIUS, Rw)  # CBF exponential gamma*hx

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

# https://matplotlib.org/stable/gallery/color/named_colors.html
color = [
    "blue",
    "green",
    "red",
    "cyan",
    "magenta",
    "yellow",
    "black",
    "pink",
    "gray",
    "aqua",
    "cyan"
]

for idx in range(0, STEPS, 20):
    robot.q = q_online[idx]
    for jdx, j in enumerate(circles):
        j.remove()
    lll = obtain_link_centres(q_online[idx].reshape(2,))
    for zdx, zz in enumerate(lll):
        x, y, z = create_sphere(Rw, zz)
        circles[zdx] = robot_fig.ax.plot_wireframe(x, y, z, label=f"{zdx}", color=color[zdx])

    robot_fig.step()

print(lll)
robot_fig.ax.legend()
robot_fig.hold()

circle1 = plt.Circle((1, 1), RADIUS, color='r')
xy_fig, xy_ax = plt.subplots()
xy_ax.axis('equal')
xy_ax.add_patch(circle1)
xy_ax.plot(x_online[:, 0], x_online[:, 1], label=f"Gamma = {GAMMA}")
xy_ax.plot(xs[:, 0], xs[:, 1], label="Offline trajectory")
xy_ax.legend()

plt.show()
