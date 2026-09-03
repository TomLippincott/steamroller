import re
import os.path
from typing import Any
from functools import partial
from SCons.Builder import Builder
from SCons.Action import Action
from .grid_engine import GridEngine
from .base_engine import BaseEngine

class LocalEngine(BaseEngine):

    name = "local"
    #parameters = {}
    
    #@property
    #def queues(self):
    #    return []

    @classmethod
    def available(cls, *argv, **argd) -> bool:
        return True

    @classmethod
    def variables(cls) -> [Any]:
        return []

    # def create_builder(self, env, builder):
    #     commands = builder.action.presub_lines(env)
    #     chdir = builder.action.chdir
    #     # m = re.match(r"^\s*(\S*[Pp]ython3?)\s+(.*?\.py)\s+(.*)$", commands[0])
    #     # if not m:
    #     #     raise Exception("Could not parse Python command: '{}'".format(commands[0]))
    #     # interpreter, script, args = m.groups()
    #     # if not os.path.exists(script):
    #     #     raise Exception("No such file: '{}'".format(script))



    #     action = Action(commands)
    #     return Builder(
    #         generator=self.create_generator(commands),
    #         #action=action, #Action(commands),
    #         #create_method(generator, chdir, self.submit_string),
    #         #    self.create_command_printer(generator),
    #         #),
    #         target_factory=lambda x : env.Value(x), #partial(self.target_factory, commands, env),
    #         emitter=self.create_emitter(commands),
    #     )
    #     return builder
