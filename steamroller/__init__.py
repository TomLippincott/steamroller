import re
import sys
import functools
from SCons.Action import Action, CommandAction
from SCons.Builder import Builder
from SCons.Script import Delete
from SCons.Variables import Variables
import SCons.Util
import subprocess
import logging
import time
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


def qsub(commands, name, std, dep_ids=[], grid_resources=[], working_dir=None, queue="all.q"):
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
    qcommand = "qsub -terse -shell n -V -N {} -q {} -b n {} {} -j y -o {} {}".format(name, queue, wd, deps, std, res)
    logging.info("\n".join(commands))
    p = subprocess.Popen(shlex.split(qcommand), stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    out, err = p.communicate("\n".join(commands).encode())
    return int(out.strip())


def LocalBuilder(env, **args):
    return Builder(**args)


def GridBuilder(env, **args): #action=None, generator=None, emitter=None, chdir=None, **args):
    action = args.get("action", None)
    queue = env["GPU_QUEUE"] if env.get("USE_GPU", False) else env["CPU_QUEUE"]
    resources = env["GPU_RESOURCES"] if env.get("USE_GPU", False) else env["CPU_RESOURCES"]
    generator = args.get("generator", None)
    emitter = args.get("emitter", None)
    chdir = args.get("chdir", None)
    if action:
        if isinstance(action, str) or isinstance(action, list) and all([isinstance(a, str) for a in action]):
            generator = lambda target, source, env, for_signature : action
        else:
            raise Exception("Only simple string actions (and lists of them) are supported!")
            
    def command_printer(target, source, env):
        command = generator(target, source, env, False)
        return ("Grid(command={}, queue={}, resources={})" if env["USE_GRID"] else "Local(command={0})").format(
            env.subst(command, target=target, source=source),
            queue,
            resources,
            chdir,
        )

    def grid_method(target, source, env):
        command = generator(target, source, env, False)
        if chdir:
            nchdir = env.Dir(chdir).abspath
        else:
            nchdir = None
        depends_on = set(filter(lambda x : x != None, [s.GetTag("built_by_job") for s in source]))
        command = env.subst(command, source=source, target=target)
        job_id = qsub(command, 
                      args.get("GRID_LABEL", env.get("GRID_LABEL", "steamroller")),
                      "{}.qout".format(target[0].abspath), 
                      depends_on,
                      resources,
                      nchdir,
                      queue,
        )
        for t in target:
            t.Tag("built_by_job", job_id)
        logging.info("Job %d depends on %s", job_id, depends_on)
        return None
    
    return Builder(action=Action(grid_method, command_printer, name="steamroller"), emitter=emitter) if env["USE_GRID"] else Builder(**args)



def generate(env):
    grid_vars = ["GPU_PREAMBLE", "GPU_RESOURCES", "GPU_QUEUE", "CPU_RESOURCES", "CPU_QUEUE", "USE_GPU", "USE_GRID"]
    env["GPU_PREAMBLE"] = "module load cuda90/toolkit"
    env["GPU_RESOURCES"] = ["h_rt=100:0:0", "gpu=1"]
    env["GPU_QUEUE"] = "gpu.q"
    env["CPU_RESOURCES"] = ["h_rt=100:0:0", "mem_free=8G"]
    env["CPU_QUEUE"] = "all.q"
    env["USE_GPU"] = env.get("USE_GPU", False)
    env["USE_GRID"] = env.get("USE_GRID", False)
    for item in sys.argv:
        toks = item.split("=")
        if len(toks) > 1 and toks[0] in grid_vars:
            name = toks[0]
            value = "=".join(toks[1:])
            if name in ["USE_GRID", "USE_GPU"]:
                env[name] = value in ["1", "True"]
            else:
                env[name] = value                
    
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
            use_gpu = env.get("USE_GPU", False)
            env["BUILDERS"][name] = GridBuilder(
                env,
                **ActionMaker(
                    env,
                    interpreter,
                    script,
                    args,
                    use_gpu=use_gpu,
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
            


def exists(env):
    return 1
