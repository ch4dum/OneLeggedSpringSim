import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from sympy.utilities.lambdify import lambdify
import tkinter as tk
from tkinter import simpledialog, messagebox

# Define parameters and previous classes/functions
class Params:
    def __init__(self):
        self.g = 9.81
        self.ground = 0.0  # Ground at y = 0
        self.l = 1
        self.m = 1
        self.r = 0.1  # meters (radius of the pulley)
        self.k = 200  # Spring stiffness
        self.theta = 5 * (np.pi / 180)  # Fixed angle
        self.c = 5    # Base damping coefficient
        self.c_gain = 2  # Gain for adjusting damping coefficient
        self.pause = 0.001
        self.fps = 100
        self.tail_length = 0.5  # Fixed tail length in meters


# Define functions for flight, stance, and other phases

def contact(t, z, m, g, l0, k, theta, *args):
    x, x_dot, y, y_dot = z
    return y - l0 * np.cos(theta)  # Detect when foot touches the ground

contact.terminal = True
contact.direction = -1

def release(t, z, m, g, l0, k, theta, *args):
    x, x_dot, y, y_dot = z
    l = np.sqrt(x**2 + y**2)
    return l - l0  # Detect when leg reaches its natural length

release.terminal = True
release.direction = +1

def apex(t, z, m, g, l0, k, theta, *args):
    x, x_dot, y, y_dot = z
    return y_dot  # Detect when vertical velocity is zero (apex)

apex.terminal = True
apex.direction = -1

def flight(t, z, m, g, l0, k, theta, drag_coefficient=0.1):
    x, x_dot, y, y_dot = z
    # Calculate drag force
    drag_force_x = -drag_coefficient * x_dot
    drag_force_y = -drag_coefficient * y_dot
    return [x_dot, drag_force_x / m, y_dot, -g + drag_force_y / m]

def stance(t, z, m, g, l0, k, theta, c, *args):
    x, x_dot, y, y_dot = z
    x_c = 0  # Fixed point
    l = np.sqrt((x - x_c)**2 + y**2)
    if l == 0:  # Prevent division by zero
        l = 1e-6
    # Accelerations
    x_dd = x_dd_func(x, y, x_dot, y_dot, x_c, m, g, k, l0)
    y_dd = y_dd_func(x, y, x_dot, y_dot, x_c, m, g, k, l0)
    # Add damping
    x_dd -= (c / m) * x_dot
    y_dd -= (c / m) * y_dot
    return [x_dot, x_dd, y_dot, y_dd]

# Function to adjust damping coefficient based on landing velocity
def adjust_damping_coefficient(landing_velocity, params):
    base_c = params.c
    c = base_c + params.c_gain * abs(landing_velocity)
    c_max = 20  # Set a realistic maximum damping value
    c = np.clip(c, base_c, c_max)
    return c

# Define the onestep function
def onestep(z0, t0, params):
    dt = 5
    x, x_d, y, y_d = z0
    m, g, k = params.m, params.g, params.k
    l0, theta = params.l, params.theta

    t_output = np.array([t0])
    z_output = np.array([[*z0, x + l0 * np.sin(theta), y - l0 * np.cos(theta)]])

    #####################################
    ###         contact phase         ###
    #####################################

    # Flight until contact
    contact_sol = solve_ivp(
        flight, [t0, t0 + dt], z0, method='Radau',
        events=contact, atol=1e-9, rtol=1e-9,
        args=(m, g, l0, k, theta)
    )

    t_contact = contact_sol.t
    z_contact = contact_sol.y.T

    # Calculate foot position for animation
    x_foot = z_contact[:, 0] + l0 * np.sin(theta)
    y_foot = z_contact[:, 2] - l0 * np.cos(theta)
    y_foot = np.maximum(y_foot, params.ground)  # Ensure foot does not go below ground

    # Append foot position into z vector
    z_contact_output = np.hstack((z_contact, x_foot.reshape(-1, 1), y_foot.reshape(-1, 1)))

    # Add to output
    t_output = np.hstack((t_output, t_contact[1:]))
    z_output = np.vstack((z_output, z_contact_output[1:, :]))

    #####################################
    ## adjust new state for next phase ##
    #####################################
    if contact_sol.status == 1 and len(contact_sol.t_events[0]) > 0:
        t0 = contact_sol.t_events[0][0]
        z0 = contact_sol.y_events[0][0]
    else:
        t0 = t_contact[-1]
        z0 = z_contact[-1]

    # Adjust damping coefficient based on landing velocity
    c = adjust_damping_coefficient(z0[3], params)

    #####################################
    ###          stance phase         ###
    #####################################

    # Stance until release
    release_sol = solve_ivp(
        stance, [t0, t0 + dt], z0, method='RK45',
        events=release, atol=1e-8, rtol=1e-8,
        args=(m, g, l0, k, theta, c)
    )

    t_release = release_sol.t
    z_release = release_sol.y.T

    # Append foot position for animation
    x_foot = z_release[:, 0] + l0 * np.sin(theta)
    y_foot = z_release[:, 2] - l0 * np.cos(theta)
    y_foot = np.maximum(y_foot, params.ground)  # Ensure foot does not go below ground
    z_release_output = np.hstack((z_release, x_foot.reshape(-1, 1), y_foot.reshape(-1, 1)))

    # Add to output
    t_output = np.hstack((t_output, t_release[1:]))
    z_output = np.vstack((z_output, z_release_output[1:, :]))

    #####################################
    ###           apex  phase         ###
    #####################################

    # Update initial conditions for apex phase
    if release_sol.status == 1 and len(release_sol.t_events[0]) > 0:
        t0 = release_sol.t_events[0][0]
        z0 = release_sol.y_events[0][0]
    else:
        t0 = t_release[-1]
        z0 = z_release[-1]

    # Flight until apex
    apex_sol = solve_ivp(
        flight, [t0, t0 + dt], z0, method='RK45',
        events=apex, atol=1e-8, rtol=1e-8,
        args=(m, g, l0, k, theta)
    )

    t_apex = apex_sol.t
    z_apex = apex_sol.y.T

    # Calculate foot position for animation
    x_foot = z_apex[:, 0] + l0 * np.sin(theta)
    y_foot = z_apex[:, 2] - l0 * np.cos(theta)
    y_foot = np.maximum(y_foot, params.ground)  # Ensure foot does not go below ground
    z_apex_output = np.hstack((z_apex, x_foot.reshape(-1, 1), y_foot.reshape(-1, 1)))

    # Add to output
    t_output = np.hstack((t_output, t_apex[1:]))
    z_output = np.vstack((z_output, z_apex_output[1:, :]))

    return z_output, t_output

# Fixed point function definition
def fixedpt(z0, params):
    t0 = 0
    z1, t1 = onestep(z0, t0, params)
    N = len(t1) - 1
    return z1[N, 0] - z0[0], z1[N, 1] - z0[1], z1[N, 2] - z0[2], z1[N, 3] - z0[3]

# Function to compute Lagrangian
def compute_lagrangian():
    global x_dd_func, y_dd_func
    x, x_c, y = sy.symbols('x x_c y')
    x_d, y_d = sy.symbols('x_d y_d')
    m, g, k, l_0 = sy.symbols('m g k l_0')
    l = sy.sqrt((x - x_c)**2 + y**2)
    T = m / 2 * (x_d**2 + y_d**2)
    V = m * g * y + k / 2 * (l - l_0)**2
    L = T - V
    x_dd, y_dd = sy.symbols('x_dd y_dd', real=True)
    q = sy.Matrix([x, y])
    q_d = sy.Matrix([x_d, y_d])
    q_dd = sy.Matrix([x_dd, y_dd])
    dL_dq_d = []
    dt_dL_dq_d = []
    dL_dq = []
    EOM = []
    for i in range(len(q)):
        dL_dq_d.append(sy.diff(L, q_d[i]))
        temp = 0
        for j in range(len(q)):
            temp += sy.diff(dL_dq_d[i], q[j]) * q_d[j] + sy.diff(dL_dq_d[i], q_d[j]) * q_dd[j]
        dt_dL_dq_d.append(temp)
        dL_dq.append(sy.diff(L, q[i]))
        EOM.append(dt_dL_dq_d[i] - dL_dq[i])
    EOM = sy.Matrix([EOM[0], EOM[1]])
    x_dd_solve = sy.solve(EOM[0], x_dd)
    y_dd_solve = sy.solve(EOM[1], y_dd)
    x_dd_func = lambdify((x, y, x_d, y_d, x_c, m, g, k, l_0), x_dd_solve[0], 'numpy')
    y_dd_func = lambdify((x, y, x_d, y_d, x_c, m, g, k, l_0), y_dd_solve[0], 'numpy')

# Function to compute the required torque
def compute_required_torque(m, g, k, l0, desired_height, r, efficiency=1):
    E_required = m * g * desired_height / efficiency  # Adjust for losses
    delta_l = np.sqrt((2 * E_required) / k)
    F = k * delta_l
    torque = F * r
    l_final = l0 - delta_l
    return torque, delta_l, l_final

# Function for multiple steps of simulation
def n_step(zstar, params, steps):
    z0 = zstar
    t0 = 0

    z = np.zeros((1, 6))
    t = np.zeros(1)

    for i in range(steps):
        z_step, t_step = onestep(z0, t0, params)
        if i == 0:
            z = z_step
            t = t_step
        else:
            z = np.vstack((z, z_step[1:, :]))
            t = np.hstack((t, t_step[1:]))

        z0 = z_step[-1, :4]
        t0 = t_step[-1]

    return z, t

def animate(z, t, params):
    data_pts = 1 / params.fps
    t_interp = np.arange(t[0], t[-1], data_pts)
    m, n = np.shape(z)
    shape = (len(t_interp), n)
    z_interp = np.zeros(shape)

    for i in range(n):
        f = interpolate.interp1d(t, z[:, i], fill_value="extrapolate")
        z_interp[:, i] = f(t_interp)

    l = params.l
    tail_length = 0.5  # Total length of the tail
    tail_angle = -30 * (np.pi / 180)  # Initial angle of the tail relative to the vertical
    tail_angle_dot = 0.0  # Initialize tail angular velocity
    k_p, k_d = 5, 0.5  # PD Controller gains

    window_width = 2 * l
    window_height = 3.0 * l

    max_x = np.max(z[:, 0])

    wall_x = None  # Removed wall_x calculation
    wall_width = None  # Removed wall_width definition

    plt.figure()
    for i in range(len(t_interp)):
        plt.cla()
        x, y = z_interp[i, 0], z_interp[i, 2]
        x_dot, y_dot = z_interp[i, 1], z_interp[i, 3]
        x_foot, y_foot = z_interp[i, 4], z_interp[i, 5]

        # Ensure foot and hip do not go below ground
        y_foot = np.maximum(y_foot, params.ground)
        y = np.maximum(y, params.ground)

        # Analyze target tail tip position
        x_target, y_target = analyze_tail_target_position(x, y, x_dot, y_dot)

        # Compute desired tail angle using inverse kinematics
        desired_theta = inverse_kinematics_tail(x, y, x_target, y_target, tail_length)

        # Calculate torque for tail using PD Controller
        torque = tail_balance_control(tail_angle, tail_angle_dot, desired_theta, 250.0, 20.0)

        # Update tail angular velocity and angle
        tail_angle_dot += torque * params.pause  # Update angular velocity
        tail_angle += tail_angle_dot * params.pause  # Update angle

        # Use the forward_kinematics_tail function
        x_tail_start, y_tail_start, x_tail_end, y_tail_end = forward_kinematics_tail(x, y, tail_angle, tail_length)

        # Removed wall drawing code
        # wall_patch = plt.Rectangle((wall_x - wall_width / 2, params.ground), wall_width, wall_height, color='blue', alpha=0.5)
        # plt.gca().add_patch(wall_patch)

        # Draw robot components
        plt.plot([x, x_foot], [y, y_foot], linewidth=2, color='black')  # Leg
        plt.plot(x, y, color='red', marker='o', markersize=10)  # Hip
        plt.plot([x_tail_start, x_tail_end], [y_tail_start, y_tail_end], linewidth=2, color='green')  # Tail visualization

        # Draw pendulums at tail ends
        pendulum_radius = 0.025
        pendulum_start = plt.Circle((x_tail_start, y_tail_start), pendulum_radius, color='blue', fill=True)
        pendulum_end = plt.Circle((x_tail_end, y_tail_end), pendulum_radius, color='blue', fill=True)
        plt.gca().add_patch(pendulum_start)
        plt.gca().add_patch(pendulum_end)

        # Draw ground line
        plt.plot([-10, 10], [params.ground, params.ground], color='brown', linewidth=2)

        window_xmin = x - window_width / 2
        window_xmax = x + window_width / 2
        window_ymin = max(y - window_height / 2, params.ground - 0.1)
        window_ymax = y + window_height / 2

        plt.xlim(window_xmin, window_xmax)
        plt.ylim(window_ymin, window_ymax)
        plt.gca().set_aspect('equal')
        plt.pause(params.pause)
        

    plt.show()

def forward_kinematics_tail(x, y, theta, tail_length):
    """
    Compute the positions of the two tail endpoints using forward kinematics.
    """
    tail_half_length = tail_length / 2
    x_start = x - tail_half_length * np.sin(theta)
    y_start = y - tail_half_length * np.cos(theta)
    x_end = x + tail_half_length * np.sin(theta)
    y_end = y + tail_half_length * np.cos(theta)
    return x_start, y_start, x_end, y_end

def tail_balance_control(theta, theta_dot, desired_theta, k_p, k_d):
    """
    PD Controller for tail angle balance.
    """
    error = desired_theta - theta
    torque = k_p * error - k_d * theta_dot
    return torque

def inverse_kinematics_tail(x, y, x_target, y_target, tail_length):
    """
    Compute the tail angle needed to reach the target tail tip position.
    """
    dx = x_target - x
    dy = y_target - y
    distance = np.hypot(dx, dy)
    max_reach = tail_length / 2  # Since the tail extends from -half_length to +half_length
    if distance > max_reach:
        # Target is unreachable, adjust to maximum reach
        scaling_factor = max_reach / distance
        dx *= scaling_factor
        dy *= scaling_factor
    elif distance < 1e-6:
        # Target is too close to the hip; default angle is zero
        return 0.0
    theta = np.arctan2(dx, dy)
    return theta

def analyze_tail_target_position(x, y, x_dot, y_dot):
    """
    Analyze the robot's state and compute the desired tail tip position (x_target, y_target)
    to help balance the robot.
    """
    # Simple heuristic: move tail in opposite direction of horizontal velocity
    k = 0.5  # Proportionality constant
    x_offset = -k * x_dot
    y_offset = 0.0  # Keep y component constant for simplicity
    x_target = x + x_offset
    y_target = y + y_offset
    return x_target, y_target

# Tkinter GUI setup
def start_simulation():
    params = Params()
    root = tk.Tk()
    root.withdraw()
    desired_height = simpledialog.askfloat("Input", "Enter the desired jump height (in meters):", minvalue=0.1, maxvalue=2.5)
    if desired_height is None:
        return

    # Efficiency factor
    efficiency = 1  # Assume 90% efficiency

    # Increase the desired height by 10% for actual jump height
    actual_jump_height = desired_height

    compute_lagrangian()
    torque, delta_l, l_final = compute_required_torque(params.m, params.g, params.k, params.l, actual_jump_height, params.r, efficiency)
    messagebox.showinfo("Required Torque and Spring Information", 
                        f"Required Torque to compress the spring: {torque:.4f} Nm\n"
                        f"Spring compression distance (delta_l): {delta_l:.4f} m\n"
                        f"Final spring length after compression (l_final): {l_final:.4f} m")

    # Update spring length in parameters
    params.l = l_final  # Use final spring length after compression

    # Calculate y_d needed to reach the actual jump height
    y_d = np.sqrt(2 * params.g * actual_jump_height / efficiency)  # Include efficiency in velocity calculation
    params.theta = 5 * (np.pi / 180)

    # Set the robot's initial state: Starting on the ground
    x = 0.0  # Horizontal position
    x_d = 0.0  # Horizontal velocity
    y = params.ground + params.l  # Vertical position: ground level + spring length
    z0 = np.array([x, x_d, y, y_d])

    # Find new equilibrium point with adjusted initial conditions
    zstar = fsolve(fixedpt, z0, args=(params,))
    print(f"zstar : {zstar}")

    # Simulation for multiple steps
    z, t = n_step(zstar, params, 3)

    # Animation
    animate(z, t, params)

# Tkinter main window
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Jump Simulation")
    root.geometry("300x150")
    label = tk.Label(root, text="Jump Simulation (with Damping)", font=("Helvetica", 12))
    label.pack(pady=10)
    start_button = tk.Button(root, text="Start Simulation", command=start_simulation, font=("Helvetica", 10))
    start_button.pack(pady=10)
    root.mainloop() 

