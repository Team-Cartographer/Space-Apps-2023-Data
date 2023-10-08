from time import time 

def timeit(method):
    def timed(*args, **kw):
        time_start = time()
        result = method(*args, **kw)
        time_end = time()
        print(f"'{method.__name__}()' executed in {time_end - time_start:.3f}s")
        return result

    return timed

class CodeNotWrittenError(Exception):
    def __init__(self, message="This code is not implemented yet"):
        self.message = message
        super().__init__(self.message)

class ObsoleteCodeError(Exception):
    def __init__(self, message="This code is old, does not work, and should not be used"):
        self.message=message
        super().__init__(self.message)

if __name__ == "__main__":
    import os
    print(os.path.exists('data//years//2016.pkl'))
    

