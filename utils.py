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


class CodeNotWrittenError(Exception):
    def __init__(self, message="This code is not implemented yet"):
        self.message = message
        super().__init__(self.message)


class NoExampleError(Exception):
    def __init__(self, message="There is no example usage for this code here"):
        self.message = message
        super().__init__(self.message)
