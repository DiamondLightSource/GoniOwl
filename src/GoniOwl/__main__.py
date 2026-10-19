from argparse import ArgumentParser

from . import __version__

import os
import cothread
from threading import Thread

import cv2
import time
from softioc import builder, softioc

from GoniOwl.GoniOwl_controller import GoniOwlController

__all__ = ["main"]

print("Hello I am setting up")
goniowl_controller = GoniOwlController()
builder.SetDeviceName("GONIOWL-TEST")
builder.stringIn("WHOAMI", initial_value="GoniOwl Python IOC")
builder.stringIn("HOSTNAME", initial_value=os.uname()[1])
goniostatus = builder.aOut("GONIOSTATUS", initial_value=0, LOPR=0, HOPR=3, PINI="YES")
inferpin = builder.boolOut("INFER", ZNAM="Disable", ONAM="Enable", HIGH=1)
#running = builder.boolOut("RUNNING", ZNAM="Disable", ONAM="Enable")
builder.LoadDatabase()
softioc.iocInit()

time.sleep(5)

def update():
    while True:
        a = inferpin.get()
        while a == 1:
            goniostatus.set(goniowl_controller.infer())
            a = 0
        cothread.Sleep(1)

cothread.Spawn(update)

softioc.interactive_ioc(globals())
