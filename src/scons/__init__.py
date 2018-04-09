import functools
from SCons.Action import Action
from SCons.Builder import Builder
import SCons.Util
import subprocess
import logging
import time
import shlex
from steamroller import data_sets

try:
    import drmaa
except:
    drmaa = False

def qsub(command, name, std, dep_ids=[], grid_resources=[]):
    deps = "" if len(dep_ids) == 0 else "-hold_jid {}".format(",".join([str(x) for x in dep_ids]))
    res = "" if len(grid_resources) == 0 else "-l {}".format(",".join([str(x) for x in grid_resources]))
    qcommand = "qsub -shell n -V -N {} -b y -cwd -j y -terse -o {} {} {} ".format(name, std, deps, res) + command
    logging.info(qcommand)
    p = subprocess.Popen(shlex.split(qcommand), stdout=subprocess.PIPE)
    out, err = p.communicate()
    return int(out.strip())


#def make_emitter(script):
#    def emitter(target, source, env):
#        return (target + [target[0].rstr() + ".resources.txt"], source)
#    return emitter


def make_builder(env, generator, emitter, name):
    #timed_command = "python -m steamroller.tools.timer ${TARGETS[-1]} -- " + base_command
    def grid_method(target, source, env):
        depends_on = set(filter(lambda x : x != None, [s.GetTag("built_by_job") for s in source]))
        job_id = qsub(env.subst(timed_command, source=source, target=target), name, "{}.qout".format(target[-1].rstr()), depends_on, env["GRID_RESOURCES"])
        for t in target:
            t.Tag("built_by_job", job_id)
        logging.info("Job %d depends on %s", job_id, depends_on)
        return None

    return Builder(generator=generator, emitter=emitter)
                   #Action(grid_method, "grid(" + base_command + ")") if env["GRID"] else Action(generator=base_command))
#emitter=make_emitter(base_command.split()[0]))

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
