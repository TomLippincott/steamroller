import functools
import re
from abc import ABC, abstractmethod
import typing
from typing import Any
import os
import os.path
import hashlib
import logging
import subprocess
import shlex
from SCons.Builder import Builder
from SCons.Action import Action, CommandAction, FunctionAction
from SCons.Script import GetOption
import SCons.Subst
from ..util import prepare_commands, check_for_executable, generate_path


logger = logging.getLogger("steamroller")


class BaseEngine(ABC):

    name = "base"
    parameters = {}

    @classmethod
    @abstractmethod
    def variables(cls) -> [Any]:
        raise NotImplemented()
    
    @classmethod
    @abstractmethod
    def available(cls, *argv, **argd) -> bool:
        raise NotImplemented()

    def create_generator(self, builder):        
        def generator(target, source, env, for_signature):
            commands = builder.action.presub_lines(env)
            return commands
        return generator
    
    def create_emitter(self, builder, commands=[]):
        overrides = {k : v for k, v in builder.overrides.items()}
        file_deps = set()
        for command in commands:
            for tok in command.split():
                if os.path.exists(tok):
                    file_deps.add(tok)
        def emitter(target, source, env):
            actual_targets = []
            # if no targets were specified (this is not an ideal way to test that)
            if all([str(s).startswith(str(t)) for t, s in zip(target, source)]):
                for t in env["STEAMROLLER_DEFAULT_TARGETS"]:
                    comps, hsh = generate_path(target, source, env, commands) #+ [t]
                    pref, suf = re.match(r"^(.*?)(\.[^\.]+(?:\.gz)?)?$", t).groups()
                    comps.append("{}_{}{}".format(pref, hsh, suf))
                    actual_targets.append(env.File(os.path.join(*comps)))
            else:
                actual_targets = [env.File(t) for t in target]
            [env.Depends(t, s) for t in actual_targets for s in file_deps]
            if "STEAMROLLER_LABEL" in env:
                env.Alias(env["STEAMROLLER_LABEL"], actual_targets)
            if env["STEAMROLLER_ALWAYS"]:
                [env.AlwaysBuild(t) for t in actual_targets]
            return (actual_targets, source)
        return emitter

    def create_builder(self, env, builder):
        retval = Builder(
            generator=self.create_generator(builder) if isinstance(builder.action, CommandAction) else None,
            target_factory=lambda x : env.Value(x),
            emitter=self.create_emitter(builder, commands=builder.action.presub_lines(env)),
            **builder.overrides
        ) if isinstance(builder.action, CommandAction) else Builder(
            action=builder.action,
            target_factory=lambda x : env.Value(x),
            emitter=self.create_emitter(builder, []),
            **builder.overrides
        )
        return retval
    
    def create_command_printer(self, generator):
        def command_printer(target, source, env):
            commands = self.prepare_commands(target, source, env, generator(target, source, env, False))
            depends_on = set(filter(lambda x : x != None, [s.GetTag("built_by_pseudo_job") for s in source]))
            job_id = self.pseudo_id
            for t in target:
                t.Tag("built_by_pseudo_job", job_id)
            self.pseudo_id += 1
            return "{0}({1})".format(                
                self.name.title(),
                ", ".join(
                    [
                        "commands={}".format(commands),
                        "name={}".format(self.create_name(target, source, env)),
                        "job_id={}".format(job_id),
                        "depends_on_jobs={}".format(depends_on),
                    ]
                )
            )
        return command_printer
    
