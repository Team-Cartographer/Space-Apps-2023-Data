import sys
import venv
from os import getcwd, path
from subprocess import run
import time as time

venv_folder = path.join(getcwd(), "subprocess_venv")
if not path.exists(venv_folder):

    start = time()

    print("Creating virtual environment")
    venv.create("subprocess_venv", with_pip=True)

    print("Installing dependencies...")

    # Add pip installation names here for any new package.
    packages = ["numpy", "torch"]
    for package in packages:
        run([sys.executable, "-m", "pip", "install", package], check=True)
        print(f"Installed package: {package}")

    print(f"Venv creation completed in {round(time()-start, 2)}s")


class CodeNotWrittenError(Exception):
    def __init__(self, message="This code is not implemented yet"):
        self.message = message
        super().__init__(self.message)


if __name__ == "__main__":
    raise CodeNotWrittenError()
  
