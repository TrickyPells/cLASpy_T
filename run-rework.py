import sys
import os
import platform
from subprocess import Popen

if platform.system() == "Windows":
    new_window_command = "cmd.exe /c start cmd.exe /c".split()

#echo1 = ['"', sys.executable, "cLASpy_T.py", "-h", "|", "pause", '"']

#process = Popen(new_window_command + echo1)

#process.wait()

os.system('cmd /c start cmd /c "{} cLASpy_T.py train -a=rf -i=Test/Orne_20130525.las & pause"'.format(sys.executable))

