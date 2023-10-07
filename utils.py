import tkinter as tk
from tkinter import messagebox
from time import time 

def timeit(method):
    def timed(*args, **kw):
        time_start = time()
        result = method(*args, **kw)
        time_end = time()
        print(f"'{method.__name__}()' executed in {time_end - time_start:.3f}s")
        return result

    return timed

def show_error(err_type, msg):
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(err_type, msg)


def show_info(title, msg):
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(title, msg)