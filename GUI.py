from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from sympy.utilities.lambdify import lambdify
import tkinter as tk
from tkinter import messagebox
import subprocess
import sys

def run_spring_damping():
    try:
        subprocess.run([sys.executable, "jump_with_damp.py"])
    except FileNotFoundError:
        messagebox.showerror("Error", "Cannot find jump_with_damp.py in the specified directory.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def run_spring_option(option):
    try:
        if option == "torque":
            subprocess.run([sys.executable, "jump_with_spring_torque.py"])
        elif option == "cross_wall":
            subprocess.run([sys.executable, "jump_with_spring_cross_wall.py"])
        elif option == "theta":
            subprocess.run([sys.executable, "jump_with_spring_theta.py"])
        elif option == "height":
            subprocess.run([sys.executable, "jump_with_spring_height.py"])
        elif option == "height_torque":
            subprocess.run([sys.executable, "jump_wth_spring_theta_torque.py"])
    except FileNotFoundError:
        messagebox.showerror("Error", f"Cannot find the specified file for the {option} option.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def spring_menu():
    spring_root = tk.Tk()
    spring_root.title("Spring Simulation Options")
    spring_root.geometry("400x400")
    label = tk.Label(spring_root, text="Select Spring Simulation Type", font=("Helvetica", 14))
    label.pack(pady=20)

    button_height = tk.Button(spring_root, text="Desired Height", font=("Helvetica", 12),
                              command=lambda: run_spring_option("height"))
    button_height.pack(pady=10)
    
    button_torque = tk.Button(spring_root, text="Desired Torque", font=("Helvetica", 12),
                              command=lambda: run_spring_option("torque"))
    button_torque.pack(pady=10)

    button_theta = tk.Button(spring_root, text="Desired Theta", font=("Helvetica", 12),
                             command=lambda: run_spring_option("theta"))
    button_theta.pack(pady=10)
    
    button_theta = tk.Button(spring_root, text="Desired Theta and Torque", font=("Helvetica", 12),
                             command=lambda: run_spring_option("height_torque"))
    button_theta.pack(pady=10)

    button_cross_wall = tk.Button(spring_root, text="Cross Wall", font=("Helvetica", 12),
                                  command=lambda: run_spring_option("cross_wall"))
    button_cross_wall.pack(pady=10)

    spring_root.mainloop()

def main_menu():
    root = tk.Tk()
    root.title("Jump Simulation")
    root.geometry("400x250")
    label = tk.Label(root, text="Select Simulation Type", font=("Helvetica", 14))
    label.pack(pady=20)

    button_damping = tk.Button(root, text="Spring with Damping", font=("Helvetica", 12),
                               command=run_spring_damping)
    button_damping.pack(pady=10)

    button_spring = tk.Button(root, text="Spring", font=("Helvetica", 12),
                              command=spring_menu)
    button_spring.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main_menu()