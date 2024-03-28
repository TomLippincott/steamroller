import re
import sys
import functools
import argparse
from SCons.Action import Action, CommandAction
from SCons.Builder import Builder
from SCons.Script import Delete
from SCons.Variables import Variables, BoolVariable, EnumVariable

import SCons
import SCons.Util
import subprocess
import logging
import shlex
import os.path
import os

from SCons.Script.Main import main as scons_main
from steamroller.engines import registry
from steamroller.environment import Environment
# Replace
# show/check progress


def main():

    engines = {}
    for name, engine_class in registry.items():
        engines[name] = engine_class().available()
    print(engines)
    
    parser = argparse.ArgumentParser()
    args, rest = parser.parse_known_args()
    
    
    #old_stdout = sys.stdout
    #old_stderr = sys.stderr
    #with open(os.devnull, "wt") as dn_ofd:
    #sys.stdout = dn_ofd
    #sys.stderr = dn_ofd
    #sys.argv[0] = "scons"
    sys.argv = ["scons"] + rest
    retval = scons_main()
    #sys.stdout = old_stdout
    #sys.stderr = old_stderr
    sys.exit(retval)










