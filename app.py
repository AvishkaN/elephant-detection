import tkinter as tk
from defection import runElephantDetection
from data_analysis import getDatInsights

def button1_action():
    runElephantDetection()
    print('detection started')

def button2_action():
    getDatInsights()
    print('started data analysis')

# Create the main window
root = tk.Tk()
root.title("Elephant Detection System")

# Set the window size (width x height)
root.geometry("450x250")  # Adjust as needed

# Create a frame for the buttons
frame = tk.Frame(root)
frame.place(relx=0.5, rely=0.5, anchor='center')

# Create Button 1
button1 = tk.Button(frame, text="Run Detector", command=button1_action)
button1.grid(row=0, column=0, padx=10, pady=10)

# Create Button 2
button2 = tk.Button(frame, text="Get Data Analysis", command=button2_action)
button2.grid(row=1, column=0, padx=10, pady=10)

# Run the application
root.mainloop()
