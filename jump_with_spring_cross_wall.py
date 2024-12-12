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
    ground_clearance = 1e-3  # Small threshold to account for numerical issues
    return y - (l0 * np.cos(theta) - ground_clearance)

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
        y_dot = 0.0
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
    """
    Simulates a single step of the robot and halts once the foot touches the ground.
    """
    dt = 5
    x, x_d, y, y_d = z0
    m, g, k = params.m, params.g, params.k
    l0, theta = params.l, params.theta

    t_output = np.zeros(1)
    t_output[0] = t0

    z_output = np.zeros((1, 6))
    z_output[0] = [*z0, x + l0 * np.sin(theta), max(y - l0 * np.cos(theta), params.ground)]

    #####################################
    ###         contact phase         ###
    #####################################
    contact.direction = -1
    contact.terminal = True

    # Simulate flight until contact with the ground
    contact_sol = solve_ivp(
        flight, [t0, t0 + dt], z0, method='RK45',
        t_eval=np.linspace(t0, t0 + dt, 201),
        dense_output=True, events=contact,
        atol=1e-10, rtol=1e-10,  # Higher precision
        args=(m, g, l0, k, theta)
    )

    # Extract flight phase data
    t_contact = contact_sol.t
    z_contact = contact_sol.y.T

    # Debugging: Check if contact event was triggered
    if contact_sol.status == 1:
        print(f"Contact event triggered at t={t_contact[-1]:.4f}, y={z_contact[-1, 2]:.4f}")
    else:
        print("Contact event NOT triggered. Adjust event function or initial conditions.")

    # Calculate foot position for animation
    x_foot = z_contact[:, 0] + l0 * np.sin(theta)
    y_foot = np.maximum(z_contact[:, 2] - l0 * np.cos(theta), params.ground)  # Ensure foot stays above ground

    # Append foot position into z vector
    z_contact_output = np.concatenate((z_contact, x_foot.reshape(-1, 1), y_foot.reshape(-1, 1)), axis=1)

    # Add to output
    t_output = np.concatenate((t_output, t_contact[1:]))
    z_output = np.concatenate((z_output, z_contact_output[1:]))

    #####################################
    ###  Halt after foot touches ground ###
    #####################################
    # After the foot touches the ground, halt the simulation by returning the state
    return z_output, t_output

# Fixed point function definition
def fixedpt(z0, params):
    t0 = 0
    z1, t1 = onestep(z0, t0, params)
    N = len(t1) - 1
    return z1[N, 0] - z0[0], z1[N, 1] - z0[1], z1[N, 2] - z0[2], z1[N, 3] - z0[3]

def compute_symbolic_jacobian():
    """
    Computes the Jacobian matrix symbolically using sympy.
    """
    # Define the symbolic variables
    x, y, theta_tail = sy.symbols('x y theta_tail')  # State variables
    x_d, y_d, theta_tail_d = sy.symbols('x_d y_d theta_tail_d')  # Velocities
    x_c = sy.symbols('x_c')  # Spring attachment point (assumed constant)
    m, g, k, l0, I_tail = sy.symbols('m g k l0 I_tail')  # System parameters
    tail_length = sy.symbols('tail_length')  # Tail length

    # State vector q and velocity vector q_d
    q = sy.Matrix([x, y, theta_tail])
    q_d = sy.Matrix([x_d, y_d, theta_tail_d])

    # Define Lagrangian as in compute_lagrangian_with_tail
    l = sy.sqrt((x - x_c)**2 + y**2)  # Spring length
    T_robot = m / 2 * (x_d**2 + y_d**2)  # Kinetic energy of robot
    T_tail = I_tail / 2 * theta_tail_d**2  # Kinetic energy of tail
    T = T_robot + T_tail
    V_robot = m * g * y + k / 2 * (l - l0)**2  # Potential energy
    L = T - V_robot  # Lagrangian

    # Compute equations of motion (EOM) using Lagrange's equations
    q_dd = sy.Matrix(sy.symbols('x_dd y_dd theta_dd'))  # Accelerations
    dL_dq_d = [sy.diff(L, q_d[i]) for i in range(3)]
    dt_dL_dq_d = [
        sum(sy.diff(dL_dq_d[i], q[j]) * q_d[j] + sy.diff(dL_dq_d[i], q_d[j]) * q_dd[j] for j in range(3))
        for i in range(3)
    ]
    dL_dq = [sy.diff(L, q[i]) for i in range(3)]
    EOM = sy.Matrix([dt_dL_dq_d[i] - dL_dq[i] for i in range(3)])

    # Define the state vector z = [x, x_d, y, y_d, theta_tail, theta_tail_d]
    z = sy.Matrix([x, x_d, y, y_d, theta_tail, theta_tail_d])
    f = sy.Matrix([
        q_d[0],  # dx/dt = x_dot
        EOM[0],  # Acceleration x_dd
        q_d[1],  # dy/dt = y_dot
        EOM[1],  # Acceleration y_dd
        q_d[2],  # dtheta/dt = theta_tail_dot
        EOM[2]   # Acceleration theta_dd
    ])

    # Compute the Jacobian symbolically
    J = f.jacobian(z)  # Jacobian of f w.r.t. z
    return J, f, z


def analyze_stability_symbolic(fixed_point, params):
    """
    Analyze stability using the symbolic Jacobian.
    """
    # Compute the symbolic Jacobian
    J_symbolic, f_symbolic, z = compute_symbolic_jacobian()

    # Create a dictionary of parameter values
    param_values = {
        'm': params.m,
        'g': params.g,
        'k': params.k,
        'l0': params.l,
        'I_tail': 0.01,  # Example moment of inertia for tail
        'tail_length': 0.5,
        'x_c': 0  # Assume the spring attachment point is fixed
    }

    # Create a dictionary of fixed-point values
    state_values = dict(zip(z, fixed_point))

    # Substitute parameter and state values into the Jacobian
    J_numeric = np.array(J_symbolic.subs({**param_values, **state_values})).astype(np.float64)

    # Compute eigenvalues and eigenvectors
    eig_values, eig_vectors = np.linalg.eig(J_numeric)
    return eig_values, eig_vectors

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
def compute_required_torque(m, g, k, l0, desired_height, r):
    E_required = m * g * desired_height
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
    tail_length = 0.5  # ความยาวทั้งหมดของหาง
    tail_angle = -30 * (np.pi / 180)  # มุมเริ่มต้นของหาง
    tail_angle_dot = 0.0  # ความเร็วเชิงมุมเริ่มต้นของหาง
    k_p, k_d = 1000, 0  # ค่ากำลังขยายของตัวควบคุม PD

    window_width = 2 * l
    window_height = 3.0 * l

    # Calculate the maximum and minimum x positions
    max_x = np.max(z[:, 0])
    min_x = np.min(z[:, 0])

    # Set the wall position to half the jump distance
    wall_x = min_x + (max_x - min_x) / 2

    wall_width = 0.05

    for i in range(len(t_interp)):
        x, y = z_interp[i, 0], z_interp[i, 2]
        x_dot, y_dot = z_interp[i, 1], z_interp[i, 3]
        x_foot, y_foot = z_interp[i, 4], z_interp[i, 5]

        # ตรวจสอบว่าขาอยู่บนพื้นและขาหดตัวหรือไม่
        leg_length = np.sqrt((x - x_foot)**2 + (y - y_foot)**2)
        if y_foot <= params.ground + 1e-3 and leg_length < l - 1e-3:
            # ปรับตำแหน่งหัวให้ขามีความยาวคงที่
            angle = np.arctan2(y - y_foot, x - x_foot)
            x = x_foot + l * np.cos(angle)
            y = y_foot + l * np.sin(angle)

        # วิเคราะห์ตำแหน่งเป้าหมายของปลายหาง
        x_target, y_target = analyze_tail_target_position(x, y, x_dot, y_dot)

        # คำนวณมุมหางที่ต้องการ
        desired_theta = inverse_kinematics_tail(x, y, x_target, y_target, tail_length)

        # คำนวณแรงบิดสำหรับหางโดยใช้ตัวควบคุม PD
        torque = tail_balance_control(tail_angle, tail_angle_dot, desired_theta, k_p, k_d)

        # อัปเดตความเร็วเชิงมุมและมุมของหาง
        tail_angle_dot += torque * params.pause
        tail_angle += tail_angle_dot * params.pause

        # ใช้ฟังก์ชัน forward_kinematics_tail
        x_tail_start, y_tail_start, x_tail_end, y_tail_end = forward_kinematics_tail(x, y, tail_angle, tail_length)

        # วาดกำแพง
        wall_patch = plt.Rectangle((wall_x - wall_width / 2, 0), wall_width, wall_height, color='blue', alpha=0.5)
        plt.gca().add_patch(wall_patch)

        # วาดส่วนประกอบของหุ่นยนต์
        # ขาวาดระหว่างหัวและเท้า
        leg, = plt.plot([x_foot, x], [y_foot, y], linewidth=2, color='black')

        # เท้าที่ตำแหน่ง (x_foot, y_foot) เป็นเครื่องหมายสีเขียวขนาดเล็ก
        foot, = plt.plot(x_foot, y_foot, color='green', marker='o', markersize=5)

        # หัวที่ตำแหน่ง (x, y) เป็นเครื่องหมายสีแดงขนาดใหญ่
        hip, = plt.plot(x, y, color='red', marker='o', markersize=10)

        # วาดหาง
        tail, = plt.plot([x_tail_start, x_tail_end], [y_tail_start, y_tail_end], linewidth=2, color='green')

        # วาดลูกตุ้มที่ปลายหาง
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

        # ลบองค์ประกอบก่อนหน้า
        hip.remove()
        foot.remove()
        leg.remove()
        tail.remove()
        pendulum_start.remove()
        pendulum_end.remove()
        wall_patch.remove()



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
    print(f"Tail positions at time step:")
    print(f"  Tail start: (x: {x_start:.4f}, y: {y_start:.4f})")
    print(f"  Tail end:   (x: {x_end:.4f}, y: {y_end:.4f})")
    return x_start, y_start, x_end, y_end


def torque_control_function(x, x_dot, y, y_dot, m, g, k, tail_angle):
    desired_angle = np.arctan2(y_dot, x_dot)  # คำนวณมุมที่ต้องการ (ตัวอย่าง: ตามอัตราเร็ว)
    torque = -k * (tail_angle - desired_angle)  # PD Controller สำหรับมุม tail
    return torque * 0.1  # ปรับค่าควบคุมให้เหมาะสม

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
    desired_height = simpledialog.askfloat("Input", "Enter the desired jump height (in meters):", minvalue=0.1, maxvalue=10.0)
    if desired_height is None:
        return

    actual_jump_height = desired_height * 1.1

    compute_lagrangian()
    torque, delta_l, l_final = compute_required_torque(params.m, params.g, params.k, params.l, actual_jump_height, params.r)
    messagebox.showinfo("Required Torque and Spring Information", 
                        f"Required Torque to compress the spring: {torque:.4f} Nm\n"
                        f"Spring compression distance (delta_l): {delta_l:.4f} m\n"
                        f"Final spring length after compression (l_final): {l_final:.4f} m")

    y_d = np.sqrt(2 * params.g * actual_jump_height)
    params.theta = 5 * (np.pi / 180)
    x, x_d, y = 0, 0.34271, 1.1
    z0 = np.array([x, x_d, y, y_d])

    # Fixed point calculation
    zstar = fsolve(fixedpt, z0, args=(params,))
    print(f"zstar : {zstar}")

    # Stability analysis using symbolic Jacobian
    eig_values, eig_vectors = analyze_stability_symbolic(zstar, params)
    print("Eigenvalues (Symbolic):", eig_values)
    print("Eigenvectors (Symbolic):\n", eig_vectors)

    # Provide feedback about stability
    stability_message = "Stable" if np.all(np.real(eig_values) < 0) else "Unstable"
    messagebox.showinfo("Stability Analysis (Symbolic)", 
                        f"Eigenvalues: {eig_values}\nSystem is {stability_message}.")

    # Simulation to calculate actual distance
    z, t = n_step(z0, params, 3)

    animate(z, t, params, wall_height=desired_height)

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
    
def compute_lagrangian_with_tail():
    global x_dd_func, y_dd_func, theta_dd_func
    # Define symbols
    x, y, theta_tail = sy.symbols('x y theta_tail')  # Position and tail angle
    x_d, y_d, theta_tail_d = sy.symbols('x_d y_d theta_tail_d')  # Velocities
    x_c = sy.symbols('x_c')  # Fixed point for the spring
    m, g, k, l0, I_tail = sy.symbols('m g k l0 I_tail')  # Parameters
    tail_length = sy.symbols('tail_length')  # Tail length
    # Spring length
    l = sy.sqrt((x - x_c)**2 + y**2)
    # Kinetic energy
    T_robot = m / 2 * (x_d**2 + y_d**2)
    T_tail = I_tail / 2 * theta_tail_d**2  # Rotational kinetic energy of tail
    T = T_robot + T_tail
    # Potential energy
    V_robot = m * g * y + k / 2 * (l - l0)**2  # Spring and gravitational potential
    V_tail = 0  # Assuming no potential energy for tail (flat plane assumption)
    V = V_robot + V_tail
    # Lagrangian
    L = T - V
    # Generalized coordinates
    q = sy.Matrix([x, y, theta_tail])
    q_d = sy.Matrix([x_d, y_d, theta_tail_d])
    q_dd = sy.Matrix(sy.symbols('x_dd y_dd theta_dd'))
    # Lagrange equations
    dL_dq_d = [sy.diff(L, q_d[i]) for i in range(3)]
    dt_dL_dq_d = [
        sum(sy.diff(dL_dq_d[i], q[j]) * q_d[j] + sy.diff(dL_dq_d[i], q_d[j]) * q_dd[j] for j in range(3))
        for i in range(3)
    ]
    dL_dq = [sy.diff(L, q[i]) for i in range(3)]
    EOM = [dt_dL_dq_d[i] - dL_dq[i] for i in range(3)]
    # Solve for accelerations
    solutions = sy.solve(EOM, q_dd)
    x_dd_func = lambdify((x, y, x_d, y_d, theta_tail, theta_tail_d, x_c, m, g, k, l0, I_tail), solutions[q_dd[0]], 'numpy')
    y_dd_func = lambdify((x, y, x_d, y_d, theta_tail, theta_tail_d, x_c, m, g, k, l0, I_tail), solutions[q_dd[1]], 'numpy')
    theta_dd_func = lambdify((x, y, x_d, y_d, theta_tail, theta_tail_d, x_c, m, g, k, l0, I_tail), solutions[q_dd[2]], 'numpy')


# Tkinter main window
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Jump Simulation")
    root.geometry("300x150")
    label = tk.Label(root, text="Jump Simulation (Cross Wall)", font=("Helvetica", 12))
    label.pack(pady=10)
    start_button = tk.Button(root, text="Start Simulation", command=start_simulation, font=("Helvetica", 10))
    start_button.pack(pady=10)
    root.mainloop()