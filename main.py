import sys
import venv
from os import getcwd, path
from subprocess import run
from utils import *

venv_folder = path.join(getcwd(), "subprocess_venv")
if not path.exists(venv_folder):

    print("Creating virtual environment")
    venv.create("subprocess_venv", with_pip=True)

    print("Installing dependencies...")

    # Add pip installation names here for any new package.
    packages = ["numpy", "torch", "torchvision"]
    for package in packages:
        run([sys.executable, "-m", "pip", "install", package], check=True)
        print(f"Installed package: {package}")

    print(f"Venv creation completed")


if __name__ == "__main__":
    raise CodeNotWrittenError

  
