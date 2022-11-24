import tkinter as tk
from tkinter import ttk
window = tk.Tk()
window.title("Welcome to TutorialsPoint")
window.geometry('800x800')
window.configure()

a = tk.Label(window, text="Radius").grid(row=0, column=0)
a1 = tk.Entry(window).grid(row=0, column=1)

b = tk.Label(window, text="Last Name").grid(row=1, column=0)
c = tk.Label(window, text="Email Id").grid(row=2, column=0)
d = tk.Label(window, text="Contact Number").grid(row=3, column=0)


b1 = tk.Entry(window).grid(row=1, column=1)
c1 = tk.Entry(window).grid(row=2, column=1)
d1 = tk.Entry(window).grid(row=3, column=1)


def clicked():
    res = "Welcome to " + tk.txt.get()
    tk.lbl.configure(text=res)


tk.btn = ttk.Button(window, text="Submit").grid(row=4, column=0)
window.mainloop()
