import re
import os.path
from SCons.Builder import Builder
from SCons.Action import Action
from .grid_engine import GridEngine


class LocalEngine(GridEngine):

    name = "local"
    parameters = {}
    
    @property
    def queues(self):
        return []

    @classmethod
    def available(cls, *argv, **argd) -> bool:
        return True

    def create_builder(self, env, builder):
        commands = builder.action.presub_lines(env)
        chdir = builder.action.chdir

        m = re.match(r"^\s*(\S*[Pp]ython3?)\s+(.*?\.py)\s+(.*)$", commands[0])
        if not m:
            raise Exception("Could not parse command: '{}'".format(commands[0]))
        
        interpreter, script, args = m.groups()
        if not os.path.exists(script):
            raise Exception("No such file: '{}'".format(script))
        #print(dir(builder))
        #builder.add_emitter(emitter=self.create_emitter(script))
        return Builder(
            action=Action(commands),
            #create_method(generator, chdir, self.submit_string),
            #    self.create_command_printer(generator),
            #),
            emitter=self.create_emitter(script),
        )
        #print(script)
        return builder
