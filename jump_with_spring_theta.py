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
        self.ground = 0.0
        self.l = 1
        self.m = 1
        self.r = 0.1  # meters (radius of the pulley)
        self.k = 200  # Spring stiffness
        self.theta = 5 * (np.pi / 180)  # Fixed angle
        self.pause = 0.001
        self.fps = 100

# Define functions for flight, stance, and other phases

def contact(t, z, m, g, l0, k, theta):
    x, x_dot, y, y_dot = z
    return y - l0 * np.cos(theta)

def release(t, z, m, g, l0, k, theta):
    x, x_dot, y, y_dot = z
    l = np.sqrt(x**2 + y**2)
    return l - l0

def apex(t, z, m, g, l0, k, theta):
    x, x_dot, y, y_dot = z
    return y_dot

def flight(t, z, m, g, l0, k, theta):
    x, x_dot, y, y_dot = z
    if y <= 0.0:
        y = 0.0
        y_dot =  0.0
        x_dot = 0.0
    return [x_dot, 0, y_dot, -g]

def stance(t, z, m, g, l0, k, theta):
    x, x_dot, y, y_dot = z
    x_c = 0  
    l = np.sqrt((x - x_c)**2 + y**2)  
    
    x_dd = x_dd_func(x, y, x_dot, y_dot, x_c, m, g, k, l0)
    y_dd = y_dd_func(x, y, x_dot, y_dot, x_c, m, g, k, l0)
    
    return [x_dot, x_dd, y_dot, y_dd]

def onestep(z0, t0, params):
    dt = 5
    x, x_d, y, y_d = z0
    m, g, k = params.m, params.g, params.k
    l0, theta = params.l, params.theta

    t_output = np.zeros(1)
    t_output[0] = t0

    z_output = np.zeros((1, 6))
    z_output[0] = [*z0, x - l0 * np.sin(theta), max(y - l0 * np.cos(theta), params.ground)]

    #####################################
    ###         contact phase         ###
    #####################################
    contact.direction = -1
    contact.terminal = True

    # Flight until contact
    contact_sol = solve_ivp(
        flight, [t0, t0 + dt], z0, method='RK45', t_eval=np.linspace(t0, t0 + dt, 201),
        dense_output=True, events=contact, atol=1e-8, rtol=1e-8,
        args=(m, g, l0, k, theta)
    )

    t_contact = contact_sol.t
    z_contact = contact_sol.y.T

    # Calculate foot position for animation
    x_foot = z_contact[:, 0] - l0 * np.sin(theta)
    y_foot = np.maximum(z_contact[:, 2] - l0 * np.cos(theta), params.ground)  # Ensure foot stays above ground

    # Append foot position into z vector
    z_contact_output = np.concatenate((z_contact, x_foot.reshape(-1, 1), y_foot.reshape(-1, 1)), axis=1)

    # Add to output
    t_output = np.concatenate((t_output, t_contact[1:]))
    z_output = np.concatenate((z_output, z_contact_output[1:]))

    #####################################
    ## adjust new state for next phase ##
    #####################################
    t0, z0 = t_contact[-1], z_contact[-1]

    #####################################
    ###          stance phase         ###
    #####################################
    release.direction = +1
    release.terminal = True

    # Stance until release
    release_sol = solve_ivp(
        stance, [t0, t0 + dt], z0, method='RK45', t_eval=np.linspace(t0, t0 + dt, 201),
        dense_output=True, events=release, atol=1e-8, rtol=1e-8,
        args=(m, g, l0, k, theta)
    )

    t_release = release_sol.t
    z_release = release_sol.y.T

    # Append foot position for animation
    x_foot = (z_release[:, 0] - l0 * np.sin(theta)).reshape(-1, 1)
    y_foot = np.maximum((z_release[:, 2] - l0 * np.cos(theta)).reshape(-1, 1), params.ground)  # Ensure foot stays above ground
    z_release_output = np.concatenate((z_release, x_foot, y_foot), axis=1)

    # Add to output
    t_output = np.concatenate((t_output, t_release[1:]))
    z_output = np.concatenate((z_output, z_release_output[1:]))

    #####################################
    ###           apex  phase         ###
    #####################################
    apex.direction = 0
    apex.terminal = True

    # Flight until apex
    apex_sol = solve_ivp(
        flight, [t0, t0 + dt], z0, method='RK45', t_eval=np.linspace(t0, t0 + dt, 201),
        dense_output=True, events=apex, atol=1e-8, rtol=1e-8,
        args=(m, g, l0, k, theta)
    )

    t_apex = apex_sol.t
    z_apex = apex_sol.y.T

    # Calculate foot position for animation
    x_foot = z_apex[:, 0] - l0 * np.sin(theta)
    y_foot = np.maximum(z_apex[:, 2] - l0 * np.cos(theta), params.ground)  # Ensure foot stays above ground
    z_apex_output = np.concatenate((z_apex, x_foot.reshape(-1, 1), y_foot.reshape(-1, 1)), axis=1)

    # Add to output
    t_output = np.concatenate((t_output, t_apex[1:]))
    z_output = np.concatenate((z_output, z_apex_output[1:]))
    
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
def compute_required_torque(m, g, k, l0, fixed_force, r):
    delta_l = np.sqrt((2 * fixed_force) / k)
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
        if i == 0:
            z, t = onestep(z0, t0, params)
        else:
            z_step, t_step = onestep(z0, t0, params)
            z = np.concatenate((z, z_step[1:]))
            t = np.concatenate((t, t_step[1:]))

        z0 = z[-1][:-2]
        t0 = t[-1]

    return z, t

def animate(z, t, params, wall_height):
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

    if wall_height < 3.1:
        wall_x = z[0, 0] + (max_x - z[0, 0]) / 3
    else:
        wall_x = z[0, 0] + (max_x - z[0, 0]) / 2

    wall_width = 0.05

    for i in range(len(t_interp)):
        x, y = z_interp[i, 0], z_interp[i, 2]
        x_dot, y_dot = z_interp[i, 1], z_interp[i, 3]
        l0 = params.l  
        
        x_foot = x - l0 * np.cos(params.theta)  
        y_foot = y - l0 * np.sin(params.theta) 

        # Analyze target tail tip position
        x_target, y_target = analyze_tail_target_position(x, y, x_dot, y_dot)

        # Compute desired tail angle using inverse kinematics
        desired_theta = inverse_kinematics_tail(x, y, x_target, y_target, tail_length)

        # Calculate torque for tail using PD Controller
        torque = tail_balance_control(tail_angle, tail_angle_dot, desired_theta, k_p, k_d)

        # Update tail angular velocity and angle
        tail_angle_dot += torque * params.pause  # Update angular velocity
        tail_angle += tail_angle_dot * params.pause  # Update angle

        # Use the forward_kinematics_tail function
        x_tail_start, y_tail_start, x_tail_end, y_tail_end = forward_kinematics_tail(x, y, tail_angle, tail_length)

        # Draw robot components
        leg, = plt.plot([x, x_foot], [y, y_foot], linewidth=2, color='black')
        hip, = plt.plot(x, y, color='red', marker='o', markersize=10)
        tail, = plt.plot([x_tail_start, x_tail_end], [y_tail_start, y_tail_end], linewidth=2, color='green')  # Tail visualization

        # Draw pendulums at tail ends
        pendulum_radius = 0.025
        pendulum_start = plt.Circle((x_tail_start, y_tail_start), pendulum_radius, color='blue', fill=True)
        pendulum_end = plt.Circle((x_tail_end, y_tail_end), pendulum_radius, color='blue', fill=True)

        plt.gca().add_patch(pendulum_start)
        plt.gca().add_patch(pendulum_end)

        window_xmin = x - window_width / 2
        window_xmax = x + window_width / 2
        window_ymin = max(y - window_height / 2, params.ground - 0.1)
        window_ymax = y + window_height / 2

        plt.xlim(window_xmin, window_xmax)
        plt.ylim(window_ymin, window_ymax)
        plt.gca().set_aspect('equal')

        plt.pause(params.pause)

        # Remove previous elements
        hip.remove()
        leg.remove()
        tail.remove()
        pendulum_start.remove()
        pendulum_end.remove()




def forward_kinematics_tail(x, y, theta, tail_length):
    """
    Compute the positions of the two tail endpoints using forward kinematics.
    
    Args:
        x (float): X position of the hip (center of the robot)
        y (float): Y position of the hip (center of the robot)
        theta (float): Tail angle (radians)
        tail_length (float): Total length of the tail
    
    Returns:
        tuple: (x_start, y_start, x_end, y_end)
    """
    tail_half_length = tail_length / 2
    x_start = x - tail_half_length * np.sin(theta)
    y_start = y - tail_half_length * np.cos(theta)
    x_end = x + tail_half_length * np.sin(theta)
    y_end = y + tail_half_length * np.cos(theta)
    
    # Add print statement to display tail positions
    # print(f"Tail positions at time step:")
    # print(f"  Tail start: (x: {x_start:.4f}, y: {y_start:.4f})")
    # print(f"  Tail end:   (x: {x_end:.4f}, y: {y_end:.4f})")
    return x_start, y_start, x_end, y_end


def torque_control_function(x, x_dot, y, y_dot, m, g, k, tail_angle):
    desired_angle = np.arctan2(y_dot, x_dot)  
    torque = -k * (tail_angle - desired_angle)  
    return torque * 0.1  

def tail_balance_control(theta, theta_dot, desired_theta, k_p, k_d):
    """
    PD Controller for tail angle balance.
    Args:
        theta: Current tail angle (radians)
        theta_dot: Current tail angular velocity (radians/sec)
        desired_theta: Desired tail angle (radians)
        k_p: Proportional gain
        k_d: Derivative gain
    Returns:
        torque: Control torque for the tail
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


# Tkinter GUI setup
def start_simulation():
    params = Params()
    root = tk.Tk()
    root.withdraw()


    fixed_force = 500.0 


    desired_theta_deg = simpledialog.askfloat("Input", "Enter the desired jump angle (Theta in degrees):", minvalue=0.0, maxvalue=90.0)
    if desired_theta_deg is None:
        return

    desired_theta_rad = np.radians(desired_theta_deg)

    force_y = fixed_force * np.sin(desired_theta_rad)  
    force_x = fixed_force * np.cos(desired_theta_rad) 
    v_y = np.sqrt((2 * force_y) / params.m)  
    v_x = np.sqrt((2 * force_x) / params.m) 
    params.theta = desired_theta_rad

    compute_lagrangian()

    x, y = 0, params.l 
    z0 = np.array([x, v_x, y, v_y])  

    zstar = fsolve(fixedpt, z0, args=(params,))
    print(f"zstar : {zstar}")

    # Simulation to calculate actual trajectory
    z, t = n_step(z0, params, 3)

    max_height = np.max(z[:, 2])
    max_distance = np.max(z[:, 0])

    messagebox.showinfo("Simulation Results",
                        f"Theta: {desired_theta_deg:.2f} degrees\n"
                        f"Maximum Height: {max_height:.2f} meters\n"
                        f"Maximum Distance: {max_distance:.2f} meters")

    # Animation and plotting
    animate(z, t, params, wall_height=1.0) 



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
    

# Tkinter main window
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Jump Simulation")
    root.geometry("300x150")
    label = tk.Label(root, text="Jump Simulation (Desired Theta)", font=("Helvetica", 12))
    label.pack(pady=10)
    start_button = tk.Button(root, text="Start Simulation", command=start_simulation, font=("Helvetica", 10))
    start_button.pack(pady=10)
    root.mainloop()