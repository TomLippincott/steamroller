import functools
from SCons.Action import Action, CommandAction
from SCons.Builder import Builder
import SCons.Util
import subprocess
import logging
import time
import shlex
from steamroller import data_sets
import os.path
import os

try:
    import drmaa
except:
    drmaa = False

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


def GridBuilder(action=None, generator=None, emitter=None, chdir=None, grid_resources=[], queue="all.q"):
    if action:
        if isinstance(action, str) or isinstance(action, list) and all([isinstance(a, str) for a in action]):
            generator = lambda target, source, env, for_signature : action
        else:
            raise Exception("Only simple string actions (and lists of them) are supported!")
            
    def command_printer(target, source, env):
        command = generator(target, source, env, False)
        return "Grid(command={}, queue={}, resources={})".format(
            env.subst(command, target=target, source=source),
            env.get("GRID_QUEUE", "all.q"),
            env.get("GRID_RESOURCES", []),
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
                      "steamroller",
                      "{}.qout".format(target[0].rstr()), 
                      depends_on,
                      env.get("GRID_RESOURCES", []),
                      nchdir,
                      env.get("GRID_QUEUE", "all.q"),
        )
        for t in target:
            t.Tag("built_by_job", job_id)
        logging.info("Job %d depends on %s", job_id, depends_on)
        return None

    return Builder(action=Action(grid_method, command_printer, name="steamroller"), emitter=emitter)


def feature_extraction_emitter(target, source, env):
    return "${WORK_PATH}/${DATA_SET_NAME}_${FEATURE_EXTRACTOR_NAME}.gz", source

def train_emitter(target, source, env):
    return "${WORK_PATH}/${DATA_SET_NAME}_${FEATURE_EXTRACTOR_NAME}_${MODEL_NAME}_model.gz", source

def apply_emitter(target, source, env):
    return "${WORK_PATH}/${DATA_SET_NAME}_${FEATURE_EXTRACTOR_NAME}_${MODEL_NAME}_probabilities.gz", source

def measurement_emitter(target, source, env):
    return "${WORK_PATH}/${DATA_SET_NAME}_${FEATURE_EXTRACTOR_NAME}_${MODEL_NAME}_${MEASUREMENT_NAME}.gz", source

def visualization_emitter(target, source, env):
    return "${WORK_PATH}/${EXPERIMENT_NAME}_${VISUALIZATION_NAME}.png", source

def generate(env):
    if not drmaa:
        logging.info("Not loading the DRMAA API for grid processing")
        if env["GRID"]:
            raise Exception("GRID=True, but unable to import DRMAA library")

    for name, (generator, emitter) in data_sets.BUILDERS.items():
        env["BUILDERS"][name] = make_builder(env, generator, emitter, name)
        
    def wait_for_grid(target, source, env):
        while True:
            p = subprocess.Popen(["qstat"], stdout=subprocess.PIPE)
            out, err = p.communicate()   
            lines = out.strip().split("\n")
            if len(lines) < 2:
                break
            else:
                counts = {}
                for line in [l for l in lines if l.startswith(" ")]:
                    toks = line.strip().split()
                    counts[toks[4][0]] = counts.get(toks[4][0], 0) + 1
            logging.info("Running: %d Waiting: %d Held: %d", counts.get("r", 0), counts.get("w", 0), counts.get("h", 0))
            time.sleep(env["GRID_CHECK_INTERVAL"])
        return None

    env["BUILDERS"]["WaitForGrid"] = Builder(action=wait_for_grid)
    
    for name, spec in env["FEATURE_EXTRACTORS"].items():
        env["BUILDERS"][name] = Builder(action=spec["COMMAND"], emitter=feature_extraction_emitter)

    for name, spec in env["MEASUREMENTS"].items():
        env["BUILDERS"][name] = Builder(action=spec["COMMAND"], emitter=measurement_emitter)

    for name, spec in env["VISUALIZATIONS"].items():
        env["BUILDERS"][name] = Builder(action=spec["COMMAND"], emitter=visualization_emitter)

    for name, spec in env["MODEL_TYPES"].items():
        env["BUILDERS"]["Train {}".format(name)] = Builder(action=spec["TRAIN_COMMAND"], emitter=train_emitter)
        env["BUILDERS"]["Apply {}".format(name)] = Builder(action=spec["APPLY_COMMAND"], emitter=apply_emitter)

def exists(env):
    return 1
