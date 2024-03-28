import re
from SCons.Environment import Base
from SCons.Variables import Variables, BoolVariable, EnumVariable
from SCons.Builder import Builder


def ActionMaker(env, interpreter, script="", args="", other_deps=[], other_args=[], emitter=lambda t, s, e : (t, s, e), chdir=None, **oargs):
    command = " ".join([x.strip() for x in [interpreter, script, args]] + ["${{'--{0} ' + str({1}) if {1} != None else ''}}".format(a.lower(), a) for a in other_args])
    before = [env["GPU_PREAMBLE"]] if oargs.get("use_gpu", False) else []
    def emitter(target, source, env):
        [env.Depends(t, s) for t in target for s in other_deps + [script]]
        return (target, source)
    return {"action" : before + [command], "emitter" : emitter, "chdir" : chdir}



def LocalBuilder(env, **args):
    return Builder(**args)

def generate(env):
    for name, builder in list(env["BUILDERS"].items()):
        commands = builder.action.presub_lines(env)
        chdir = builder.action.chdir
        #print(builder.overrides)
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
                ),
                grid_label=os.path.splitext(os.path.basename(script))[0],
                overrides=builder.overrides
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
            ("GRID_LABEL", "Name that will be assigned to job", None),
        )
        tools = argd.pop("tools", [])
        super(Environment, self).__init__(
            *argv,
            **argd,
            tools=tools + [generate],
            variables=vars
        )
        self.Decider("timestamp-newer")
