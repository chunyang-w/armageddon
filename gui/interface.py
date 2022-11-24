import tkinter as tk
from tkinter import ttk
window = tk.Tk()
window.title("Welcome to TutorialsPoint")
window.geometry('800x800')
window.configure()

a = tk.Label(window, text="Radius").grid(row=0, column=0)
radius = tk.Entry(window).grid(row=0, column=1)

b = tk.Label(window, text="angle").grid(row=1, column=0)
angle = tk.Entry(window).grid(row=1, column=1)

c = tk.Label(window, text="strength").grid(row=2, column=0)
strength = tk.Entry(window).grid(row=2, column=1)

d = tk.Label(window, text="velocity").grid(row=3, column=0)
velocity = tk.Entry(window).grid(row=3, column=1)

lat = tk.Label(window, text="latitude").grid(row=4, column=0)
velocity = tk.Entry(window).grid(row=4, column=1)

lon = tk.Label(window, text="longitude").grid(row=5, column=0)
velocity = tk.Entry(window).grid(row=5, column=1)

bearing = tk.Label(window, text="bearing").grid(row=6, column=0)
velocity = tk.Entry(window).grid(row=6, column=1)

pressure = tk.Label(window, text="pressure").grid(row=7, column=0)
velocity = tk.Entry(window).grid(row=7, column=1)


def clicked():
    res = "Welcome to " + tk.txt.get()
    tk.lbl.configure(text=res)


tk.btn = ttk.Button(window, text="Submit").grid(row=4, column=0)
window.mainloop()
