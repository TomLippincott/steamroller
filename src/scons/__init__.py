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
    qcommand = "qsub -v PATH -v PYTHONPATH -N {} -b y -cwd -j y -terse -o {} {} {} ".format(name, std, deps, res) + command
    logging.info(qcommand)
    p = subprocess.Popen(shlex.split(qcommand), stdout=subprocess.PIPE)
    out, err = p.communicate()
    return int(out.strip())


def make_emitter(script):
    def emitter(target, source, env):
        #[env.Depends(t, script) for t in target]
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
            ("CreateSplit", "python -m steamroller.tools.split --total_file ${SOURCES[0]} --train_count ${TRAIN_COUNT} --test_count ${TEST_COUNT} --train ${TARGETS[0]} --test ${TARGETS[1]}"),
            ("Evaluate", "python -m steamroller.metrics.fscore -o ${TARGETS[0]} ${SOURCES}"),
            ("CollateResources", "python -m steamroller.tools.resources -o ${TARGETS[0]} ${SOURCES}"),
            ("ModelSizes", "python -m steamroller.tools.model_sizes -o ${TARGETS[0]} ${SOURCES}"),        
            ("Plot", "python -m steamroller.plots.whisker -o ${TARGETS[0]} -f \"${FIELD}\" -t \"'${TITLE}'\" ${SOURCES}"),
    ]:
        env["BUILDERS"][name] = make_builder(env, command, name)
        
    for model in env["MODELS"]:
        env["BUILDERS"]["Train{}".format(model["name"])] = make_builder(env, model["train_command"], "Train{}".format(model["name"]))
        env["BUILDERS"]["Apply{}".format(model["name"])] = make_builder(env, model["apply_command"], "Apply{}".format(model["name"]))

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
