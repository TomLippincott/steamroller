import sys
import os
import os.path
import re
import hashlib
import json
import logging
from SCons.Environment import Base
from SCons.Variables import Variables, BoolVariable, EnumVariable, ListVariable
from SCons.Builder import Builder
from SCons.Script.Main import AddOption, GetOption
from steamroller.engines import registry
from steamroller.util import expand_experiment


logger = logging.getLogger("steamroller")


def save_config_emitter(target, source, env):
    print(target[0])
    new_targets = env.subst("work/${STEAMROLLER_CONFIG_DIGEST}/") + target[0].rstr()
    env.Alias("config", new_targets)
    return (new_targets, source)


def save_config(target, source, env):
    j = json.loads(env.Dump(format="json"))
    j["STEAMROLLER_OUTPUTS"] = [s.rstr() for s in source]
    with open(target[0].rstr(), "wt") as ofd:
        ofd.write(json.dumps(j))


def perform_experiments(env, steps):
    for experiment in env["STEAMROLLER_EXPERIMENTS"]:
        for config in expand_experiment(list(sorted(experiment.items()))):
            conf_env = env.Override(config)
            for fold in range(conf_env["STEAMROLLER_FOLDS"]):
                m = hashlib.md5()
                m.update(bytes(str(config).encode("utf-8")))
                config["STEAMROLLER_FOLD"] = fold
                config["STEAMROLLER_CONFIG_DIGEST"] = m.hexdigest()                
                fold_env = conf_env.Override(config)
                steps(fold_env)


def generate(env):
    engine = registry[env["STEAMROLLER_ENGINE"]]()
    env["BUILDERS"]["SaveConfig"] = Builder(action=save_config, STEAMROLLER_DEFAULT_TARGETS=["config.json.gz"], STEAMROLLER_LABEL="config")
    for name, builder in list(env["BUILDERS"].items()):
        env["BUILDERS"][name] = engine.create_builder(
            env,
            builder,
        )
    env.AddMethod(perform_experiments, "PerformExperiments")


def noop(targets=[], sources=[], **argd):
    return targets


class Environment(Base):

    def __init__(self, *argv, **argd):

        logging.basicConfig(level=logging.DEBUG)

        vars = argd.pop("variables")
        builders = argd.pop("BUILDERS", {})

        engines = {}
        for engine_name, engine_class in registry.items():
            engines[engine_name] = engine_class().available()

        var_list = [
            # These are meant to be set and overridden, e.g. in a 'custom.py' file:
            EnumVariable("STEAMROLLER_ENGINE", "Which steamroller engine to use", "local", [e for e, a in engines.items() if a]),
            ("STEAMROLLER_ALWAYS", "", False),
            ("STEAMROLLER_EXPERIMENTS", "", []),
            ("STEAMROLLER_FOLDS", "", 1),
            ("STEAMROLLER_AXES", "", None),
            ("STEAMROLLER_HASHING", "", False),
            
            # This can, but probably shouldn't, be overridden:
            ("STEAMROLLER_OUTPUT_ROOT", "", "work"),
            ListVariable("STEAMROLLER_NOOPS", help="Treat the specified build rules as no-ops/passthroughs.", default=[], names=list(builders.keys())),
                        
            # These are meant to be overridden at Builder definition in the SConstruct file
            ("STEAMROLLER_DEFAULT_TARGETS", "", []),
            ("STEAMROLLER_LABEL", "", None),
            
            # These are not meant to be overridden, but are used dynamically by SteamRoller
            ("STEAMROLLER_FOLD", "", 0),
            
            # These are standard SCons variables that (may) need be set a particular way for SteamRoller
            #("IMPLICIT_COMMAND_DEPENDENCIES", "Don't ", False),
        ]

        for engine_name in [engine_name for engine_name, available in engines.items() if available]:
            var_list += registry[engine_name]().variables()
        
        vars.AddVariables(*var_list)
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

        
