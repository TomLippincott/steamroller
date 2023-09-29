import re
import sys
import functools
from SCons.Action import Action, CommandAction
from SCons.Builder import Builder
from SCons.Script import Delete
from SCons.Variables import Variables, BoolVariable, EnumVariable
from SCons.Environment import Base
import SCons
import SCons.Util
import subprocess
import logging
#import time
import shlex
import os.path
import os


def ActionMaker(env, interpreter, script="", args="", other_deps=[], other_args=[], emitter=lambda t, s, e : (t, s, e), chdir=None, **oargs):
    command = " ".join([x.strip() for x in [interpreter, script, args]] + ["${{'--{0} ' + str({1}) if {1} != None else ''}}".format(a.lower(), a) for a in other_args])
    before = [env["GPU_PREAMBLE"]] if oargs.get("use_gpu", False) else []
    def emitter(target, source, env):
        [env.Depends(t, s) for t in target for s in other_deps + [script]]
        return (target, source)
    return {"action" : before + [command], "emitter" : emitter, "chdir" : chdir}


def AddBuilder(env, name, script, args, other_deps=[], interpreter="python", use_gpu=False, chdir=None):
    env.Append(
        BUILDERS={
            name : env.Builder(
                **env.ActionMaker(
                    interpreter,
                    script,
                    args,
                    other_deps=other_deps,
                    use_gpu=use_gpu,
                    chdir=chdir
                )
            )
        }
    )
    return getattr(env, name)


#def univa(commands, name, std, dep_ids=[], grid_resources=[], working_dir=None, queue="all.q"):
def univa(commands, name, std, dep_ids=[], working_dir=None, gpu_count=0, time="48:00:00", memory="8G"):
    if not isinstance(commands, list):
        commands = [commands]
    if os.path.exists(std):
        try:
            os.remove(std)
        except:
            pass
    deps = "" if len(dep_ids) == 0 else "-hold_jid {}".format(",".join([str(x) for x in dep_ids]))
    res = "" if len(grid_resources) == 0 else "-l {}".format(",".join([str(x) for x in grid_resources]))
    wd = "-wd {}".format(working_dir) if working_dir else "-cwd"
    qcommand = "qsub -terse -shell n -V -N {} -q {} -b n {} {} -j y -o {} -l h_rt={},mem_free={}".format(name, queue, wd, deps, std, time, memory)
    logging.info("\n".join(commands))
    p = subprocess.Popen(shlex.split(qcommand), stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    out, err = p.communicate("\n".join(commands).encode())
    return int(out.strip())

def slurm(commands, name, std, dep_ids=[], working_dir=None, gpu_count=0, time="48:00:00", memory="8G"):
    if not isinstance(commands, list):
        commands = [commands]
    if os.path.exists(std):
        try:
            os.remove(std)
        except:
            pass
    deps = "" if len(dep_ids) == 0 else "-d afterok:{}".format(":".join([str(x) for x in dep_ids]))
    wd = "-D {}".format(working_dir) if working_dir else "" #"-cwd"
    qcommand = "sbatch {wd} {deps} -J {name} --kill-on-invalid-dep=yes --mail-type=NONE --mem={memory} -o {std} --parsable -t {time}".format(
        name=name,
        deps=deps,
        wd=wd,
        std=std,
        time=time,
        memory=memory,
    )
    logging.info("\n".join(commands))
    commands = ["#!/bin/bash"] + commands
    p = subprocess.Popen(shlex.split(qcommand), stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    out, err = p.communicate("\n".join(commands).encode())
    return int(out.strip())


submit_commands = {
    "slurm" : slurm,
    "univa" : univa
}

def LocalBuilder(env, **args):
    return Builder(**args)

def prepare_commands(target, source, env, commands):
    escape = env.get('ESCAPE', lambda x: x)
    escape_list = SCons.Subst.escape_list
    cmd_listsA = [env.subst_list(c, SCons.Subst.SUBST_CMD, target=target, source=source) for c in commands]
    cmd_listsB = [escape_list(c[0], escape) for c in cmd_listsA]
    return [' '.join(c) for c in cmd_listsB]


def GridAwareBuilder(env, **args):
    action = args.get("action", None)
    emitter = args.get("emitter", None)
    chdir = args.get("chdir", None)
    if action:
        if isinstance(action, str) or isinstance(action, list) and all([isinstance(a, str) for a in action]):
            generator = lambda target, source, env, for_signature : action
        else:
            raise Exception("Only simple string actions (and lists of them) are supported!")
            
    def command_printer(target, source, env):
        commands = prepare_commands(target, source, env, generator(target, source, env, False))
        return ("Grid(command={command}, memory={memory}, gpu_count={gpu_count}, queue={queue}, time={time})" if env.get("USE_GRID") else "Local(command={command})").format(
            command=commands,
            memory=env.get("GRID_MEMORY"),
            gpu_count=env.get("GRID_GPU_COUNT", 0),
            queue=env.get("GRID_GPU_QUEUE") if env.get("GRID_GPU_COUNT") else env.get("GRID_CPU_QUEUE"),
            time=env.get("GRID_TIME"),
            chdir=chdir,
        )
        
    def grid_aware_method(target, source, env):
        commands = prepare_commands(target, source, env, generator(target, source, env, False))
        if chdir:
            nchdir = env.Dir(chdir).abspath
        else:
            nchdir = None
        depends_on = set(filter(lambda x : x != None, [s.GetTag("built_by_job") for s in source]))
        job_id = 1
        job_id = submit_commands[env["GRID_TYPE"]](
            commands, 
            args.get("GRID_LABEL", env.get("GRID_LABEL", "steamroller")),
            #target[-1].abspath,
            "{}.log".format(target[0].abspath), 
            depends_on,
            gpu_count=env["GRID_GPU_COUNT"],
            #resources,
            working_dir=nchdir,
            memory=env["GRID_MEMORY"],
        )
        for t in target:
            t.Tag("built_by_job", job_id)
        logging.info("Job %d depends on %s", job_id, depends_on)
        return None
    
    return Builder(
        action=Action(
            grid_aware_method,
            command_printer,
            name="steamroller"
        ),
        emitter=emitter
    )



def generate(env):
    for name, builder in list(env["BUILDERS"].items()):
        commands = builder.action.presub_lines(env)
        chdir = builder.action.chdir
        if len(commands) != 1:
            raise Exception("Steamroller only supports single-command actions")
        m = re.match(r"^\s*(\S*[Pp]ython3?)\s+(.*?\.py)\s+(.*)$", commands[0])
        if not m:
            raise Exception("Could not parse command: '{}'".format(commands[0]))
        interpreter, script, args = m.groups()
        if env["USE_GRID"]:
            env["BUILDERS"][name] = GridAwareBuilder(
                env,
                **ActionMaker(
                    env,
                    interpreter,
                    script,
                    args,
                    chdir=chdir,
                )
            )
        else:
           env["BUILDERS"][name] = LocalBuilder(
               env,
               **ActionMaker(
                   env,
                   interpreter,
                   script,
                   args,
                   chdir=chdir,
               )
           )
        
class Environment(Base):
    def __init__(self, *argv, **argd):
        vars = argd.pop("variables")
        vars.AddVariables(
            BoolVariable("USE_GRID", "", False),
            EnumVariable("GRID_TYPE", "Grid software to use", "slurm", ["slurm", "univa"]),
            ("GRID_CPU_QUEUE", "Queue/partition name for CPU tasks", "defq"),
            ("GRID_GPU_QUEUE", "Queue/partition name for GPU tasks", "a100"),
            ("GRID_GPU_PREAMBLE", "Commands to run before a GPU task (e.g. to initialize CUDA)", ""),
            ("GRID_GPU_COUNT", "How many GPUs are needed", 0),
            ("GRID_MEMORY", "How much memory to request", "8G"),
            ("GRID_TIME", "How much time to request", "06:00"),
        )
        tools = argd.pop("tools")
        super(Environment, self).__init__(
            *argv,
            **argd,
            tools=tools + [generate],
            variables=vars
        )

def exists(env):
    return 1
