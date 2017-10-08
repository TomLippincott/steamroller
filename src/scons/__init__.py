import functools
from SCons.Action import Action
from SCons.Builder import Builder
import SCons.Util
import subprocess
import logging
import time
import shlex


def qsub(command, name, std, dep_ids=[], grid_resources=[]):
    deps = "" if len(dep_ids) == 0 else "-hold_jid {}".format(",".join([str(x) for x in dep_ids]))
    res = "" if len(grid_resources) == 0 else "-l {}".format(",".join([str(x) for x in grid_resources]))
    qcommand = "qsub -shell n -V -N {} -b y -cwd -j y -terse -o {} {} {} ".format(name, std, deps, res) + command
    logging.info(qcommand)
    p = subprocess.Popen(shlex.split(qcommand), stdout=subprocess.PIPE)
    out, err = p.communicate()
    return int(out.strip())


def make_emitter(script):
    def emitter(target, source, env):
        return (target + [target[0].rstr() + ".resources.txt"], source)
    return emitter


def make_builder(env, base_command, name):
    timed_command = "/usr/bin/time --verbose -o ${TARGETS[-1]} " + base_command
    def grid_method(target, source, env):
        depends_on = set(filter(lambda x : x != None, [s.GetTag("built_by_job") for s in source]))
        job_id = qsub(env.subst(timed_command, source=source, target=target), name, "{}.qout".format(target[-1].rstr()), depends_on, env["GRID_RESOURCES"])
        for t in target:
            t.Tag("built_by_job", job_id)
        logging.info("Job %d depends on %s", job_id, depends_on)
        return None

    return Builder(action=Action(grid_method, "grid(" + base_command + ")") if env["GRID"] else Action(timed_command, base_command),
                   emitter=make_emitter(base_command.split()[0]))


def generate(env):

    env.SetDefault(
        MAX_NGRAM=4,
    )

    for name, command in [
            ("GetCount", "python -m steamroller.tools.count --input ${SOURCES[0]} --output ${TARGETS[0]}"),
            ("CreateSplit", "python -m steamroller.tools.split --total_file ${SOURCES[0]} --training_size ${TRAINING_SIZE} --testing_size ${TESTING_SIZE} --train ${TARGETS[0]} --test ${TARGETS[1]}"),
            ("NoSplit", "python -m steamroller.tools.nosplit -i ${SOURCES[0]} -o ${TARGETS[0]}"),
            ("Accuracy", "python -m steamroller.metrics.accuracy -o ${TARGETS[0]} ${SOURCES}"),
            ("FScore", "python -m steamroller.metrics.fscore -o ${TARGETS[0]} ${SOURCES}"),            
            ("CollateResources", "python -m steamroller.tools.resources -o ${TARGETS[0]} -s ${STAGE} ${SOURCES}"),
            ("CombineCSVs", "python -m steamroller.tools.combine_csvs -o ${TARGETS[0]} ${SOURCES}"),
            ("ModelSizes", "python -m steamroller.tools.model_sizes -o ${TARGETS[0]} ${SOURCES}"),        
            ("Plot", "python -m steamroller.plots.${TYPE} --output ${TARGETS[0]} --x ${X} --y ${Y} --xlabel \"${XLABEL}\" --ylabel \"${YLABEL}\" --title \"'${TITLE}'\" --input ${SOURCES[0]} --color \"${COLOR}\" --color_label \"'${COLOR_LABEL}'\""),
    ]:
        env["BUILDERS"][name] = make_builder(env, command, name)
        
    for model in env["MODELS"]:
        if not model.get("DISABLED", False):
            env["BUILDERS"]["Train{}".format(model["NAME"])] = make_builder(env, model["TRAIN_COMMAND"], "Train{}".format(model["NAME"]))
            env["BUILDERS"]["Apply{}".format(model["NAME"])] = make_builder(env, model["APPLY_COMMAND"], "Apply{}".format(model["NAME"]))

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


def exists(env):
    return 1
