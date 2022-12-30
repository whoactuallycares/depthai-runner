import platform
import os

if platform.system() != "Linux":
  exit(0)

os.system("echo 'SUBSYSTEM==\"usb\", ATTRS{idVendor}==\"03e7\", MODE=\"0666\"' | sudo tee /etc/udev/rules.d/80-movidius.rules")
os.system("pkexec udevadm control --reload-rules && sudo udevadm trigger")