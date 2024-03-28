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
    overrides = args.get("overrides", {})
    grid_label = args.get("grid_label", "steamroller")
    grid_label = env["GRID_LABEL"] if env.get("GRID_LABEL") else grid_label
    if action:
        if isinstance(action, str) or isinstance(action, list) and all([isinstance(a, str) for a in action]):
            generator = lambda target, source, env, for_signature : action
        else:
            raise Exception("Only simple string actions (and lists of them) are supported!")
            
    def command_printer(target, source, env):
        commands = prepare_commands(target, source, env, generator(target, source, env, False))
        return ("Grid(command={command}, memory={memory}, gpu_count={gpu_count}, queue={queue}, time={time}, label={label})" if env.get("USE_GRID") else "Local(command={command})").format(
            command=commands,
            memory=env.get("GRID_MEMORY"),
            gpu_count=env.get("GRID_GPU_COUNT", 0),
            queue=env.get("GRID_GPU_QUEUE") if env.get("GRID_GPU_COUNT") else env.get("GRID_CPU_QUEUE"),
            time=env.get("GRID_TIME"),
            account=env.get("GRID_ACCOUNT"),
            chdir=chdir,
            label=grid_label
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
            grid_label,
            #target[-1].abspath,
            "{}.log".format(target[0].abspath), 
            depends_on,
            gpu_count=env["GRID_GPU_COUNT"],
            #resources,
            working_dir=nchdir,
            memory=env["GRID_MEMORY"],
            queue=env.get("GRID_QUEUE", None),
            account=env.get("GRID_ACCOUNT", None)
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
        emitter=emitter,
        **overrides
    )



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
