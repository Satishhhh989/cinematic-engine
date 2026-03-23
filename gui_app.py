import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os
import sys

import subprocess

# We don't import engine directly anymore to avoid dependency errors (like cv2)
# when the GUI is run outside the virtual environment. We will call it via subprocess.

def select_input_file():
    filepath = filedialog.askopenfilename(
        title="Select Input Video",
        filetypes=(("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*"))
    )
    if filepath:
        input_var.set(filepath)
        # Auto-set output file if it's empty
        if not output_var.get():
            base, ext = os.path.splitext(filepath)
            output_var.set(base + "_hud_output.mp4")

def select_output_file():
    filepath = filedialog.asksaveasfilename(
        title="Select Save Location",
        defaultextension=".mp4",
        filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*"))
    )
    if filepath:
        output_var.set(filepath)

def run_conversion():
    input_video = input_var.get()
    output_video = output_var.get()

    if not input_video or not os.path.exists(input_video):
        messagebox.showerror("Error", "Please select a valid input video.")
        return
    if not output_video:
        messagebox.showerror("Error", "Please select an output file location.")
        return

    # Disable buttons during conversion
    btn_convert.config(state=tk.DISABLED)
    btn_input.config(state=tk.DISABLED)
    btn_output.config(state=tk.DISABLED)
    status_var.set("Status: Converting... Please wait.")

    def conversion_thread():
        try:
            # We must use the venv's python to ensure all dependencies like cv2 are available
            venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "bin", "python")
            
            if not os.path.exists(venv_python):
                # Fallback to current executable if venv is missing
                venv_python = sys.executable

            # Create a small script string to run the engine
            script_code = f"""
import sys
try:
    from engine import run_engine
    run_engine('{input_video}', '{output_video}')
except Exception as e:
    print(e, file=sys.stderr)
    sys.exit(1)
"""
            
            # Call engine via subprocess
            result = subprocess.run(
                [venv_python, "-c", script_code],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                status_var.set(f"Status: Conversion completed successfully!\nSaved to: {output_video}")
                messagebox.showinfo("Success", "Video conversion completed!")
            else:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error occurred."
                status_var.set("Status: Error occurred.")
                messagebox.showerror("Error", f"An error occurred during conversion:\n{error_msg}")
        except Exception as e:
            status_var.set("Status: Error occurred.")
            messagebox.showerror("Error", f"An unexpected error occurred:\n{str(e)}")
        finally:
            btn_convert.config(state=tk.NORMAL)
            btn_input.config(state=tk.NORMAL)
            btn_output.config(state=tk.NORMAL)

    thread = threading.Thread(target=conversion_thread)
    thread.daemon = True
    thread.start()

# Setup Tkinter window
root = tk.Tk()
root.title("Cinematic HUD Video Converter")
root.geometry("600x320")
root.resizable(False, False)

# Main frame
frame = tk.Frame(root, padx=20, pady=20)
frame.pack(expand=True, fill=tk.BOTH)

# Title
lbl_title = tk.Label(frame, text="Cinematic HUD Video Converter", font=("Arial", 16, "bold"))
lbl_title.pack(pady=(0, 20))

# Input Section
input_frame = tk.Frame(frame)
input_frame.pack(fill=tk.X, pady=10)
tk.Label(input_frame, text="Input Video: ", width=12, anchor="w").pack(side=tk.LEFT)
input_var = tk.StringVar()
tk.Entry(input_frame, textvariable=input_var, state="readonly", width=42).pack(side=tk.LEFT, padx=5)
btn_input = tk.Button(input_frame, text="Browse", command=select_input_file)
btn_input.pack(side=tk.LEFT)

# Output Section
output_frame = tk.Frame(frame)
output_frame.pack(fill=tk.X, pady=10)
tk.Label(output_frame, text="Save As: ", width=12, anchor="w").pack(side=tk.LEFT)
output_var = tk.StringVar()
tk.Entry(output_frame, textvariable=output_var, state="readonly", width=42).pack(side=tk.LEFT, padx=5)
btn_output = tk.Button(output_frame, text="Browse", command=select_output_file)
btn_output.pack(side=tk.LEFT)

# Convert Button
btn_convert = tk.Button(frame, text="Convert Video", command=run_conversion, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), pady=8, padx=15)
btn_convert.pack(pady=20)

# Status Label
status_var = tk.StringVar()
status_var.set("Status: Ready")
lbl_status = tk.Label(frame, textvariable=status_var, font=("Arial", 10), fg="#555555", justify=tk.CENTER)
lbl_status.pack()

root.mainloop()
