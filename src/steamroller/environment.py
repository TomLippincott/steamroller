import sys
import os
import os.path
import re
import logging
from SCons.Environment import Base
from SCons.Variables import Variables, BoolVariable, EnumVariable, ListVariable
from SCons.Builder import Builder
from SCons.Script.Main import AddOption, GetOption
from steamroller.engines import registry


logger = logging.getLogger("steamroller")


def generate(env):
    engine = registry[env["STEAMROLLER_ENGINE"]]()
    for name, builder in list(env["BUILDERS"].items()):
        env["BUILDERS"][name] = engine.create_builder(
            env,
            builder,
        )

def noop(targets=[], sources=[], **argd):
    return targets

class Environment(Base):
    def __init__(self, *argv, **argd):

        logging.basicConfig(level=logging.DEBUG)

        vars = argd.pop("variables")
        builders = argd.pop("BUILDERS", {})
        
        engines = {}
        for name, engine_class in registry.items():
            engines[name] = engine_class().available()

        vars.AddVariables(
            EnumVariable("STEAMROLLER_ENGINE", "Which steamroller engine to use", "local", [e for e, a in engines.items() if a]),
            ("STEAMROLLER_QUEUE", "", None),
            ("STEAMROLLER_ACCOUNT", "", None),
            ("STEAMROLLER_MEMORY", "", "8G"),
            ("STEAMROLLER_TIME", "", "06:00:00"),
            ("STEAMROLLER_GPU_COUNT", "", 0),
            ("STEAMROLLER_NAME_PREFIX", "", "steamroller"),
            ("STEAMROLLER_SUBMIT_COMMAND", "", None),
            ("STEAMROLLER_SHELL", "", "#!/bin/bash"),
            ListVariable("STEAMROLLER_NOOPS", help="Treat the specified build rules as no-ops/passthroughs.", default=[], names=list(builders.keys()))
        )
        tools = argd.pop("tools", [])
        
        super(Environment, self).__init__(
            *argv,
            **argd,
            tools=tools + [generate],
            variables=vars,
            ENV=os.environ,
            BUILDERS=builders
        )
        self.Decider("timestamp-newer")

        for builder_name in list(self["STEAMROLLER_NOOPS"]):
            setattr(self, builder_name, noop)

        
