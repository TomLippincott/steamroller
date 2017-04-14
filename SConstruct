import os
import os.path
import logging
import random
import subprocess
import shlex


vars = Variables("custom.py")
vars.AddVariables(
    ("OUTPUT_WIDTH", "Upper limit on how long a debugging line will be before it's truncated", 80),
    ("DEFAULTS", "General variables (potentially overridden by models and tasks)", {}),
    ("MODELS", "Classification models to compare", []),
    ("TASKS", "Classification tasks", []),
    ("TEST_COUNT", "Classification tasks", 10000),
    BoolVariable("GRID", "Do we have access to a grid via the qsub command?", False),
)


def print_cmd_line(s, target, source, env):
    if len(s) > int(env["OUTPUT_WIDTH"]):
        print s[:int(env["OUTPUT_WIDTH"]) - 10] + "..." + s[-7:]
    else:
        print s


env = Environment(variables=vars, ENV=os.environ, TARFLAGS="-c -z", TARSUFFIX=".tgz",
                  tools=["default"],
)


logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)


def qsub(command, dep_ids=[], resources=[]):
    deps = "" if len(dep_ids) == 0 else "-hold_jid %s" % (",".join([str(x) for x in dep_ids]))
    res = "" if len(resources) == 0 else "-l %s" % (",".join([str(x) for x in resources]))
    qcommand = "qsub -v PATH -v PYTHONPATH -b y -cwd -j y -terse -o output.out %s %s %s" % (deps, res, command)
    p = subprocess.Popen(shlex.split(qcommand), stdout=subprocess.PIPE)
    out, err = p.communicate()
    return int(out.strip())


def GridWrapper(command, resources=[]): #["qname=all.q"]):
    #command = "/usr/bin/time -f '${TARGETS[0]} %e %M' " + command
    if env["GRID"] and isinstance(command, basestring):
        def grid_method(target, source, env):
            depends_on = set(filter(lambda x : x != None, [s.GetTag("built_by_job") for s in source]))
            job_id = qsub(env.subst(command, source=source, target=target), depends_on, resources)
            for t in target:
                t.Tag("built_by_job", job_id)
            print "Job %d depends on %s" % (job_id, depends_on)
            return None
        return grid_method
    else:
        return command


defaults = env["DEFAULTS"]


for name, command in [
       ("GetCount", "python tools/get_count.py --input ${SOURCES[0]} --output ${TARGETS[0]}"),
       ("CreateSplit", "python tools/create_split.py --total_file ${SOURCES[0]} --train_count ${SOURCES[1].read()} --test_count ${TEST_COUNT} --train ${TARGETS[0]} --test ${TARGETS[1]}"),
       ("Evaluate", "python tools/evaluate.py -o ${TARGETS[0]} ${SOURCES}"),
       ("Compare", "python tools/compare.py -o ${TARGETS[0]} ${SOURCES}"),
       ("Plot", "python tools/plot.py -o ${TARGETS[0]} ${SOURCES}"),
       ]:
   env["BUILDERS"][name] = Builder(action=GridWrapper(command % defaults))
   

for model in env["MODELS"]:
    env["BUILDERS"]["Train %s" % model["name"]] = Builder(action=GridWrapper(model["train_command"] % defaults))
    env["BUILDERS"]["Apply %s" % model["name"]] = Builder(action=GridWrapper(model["apply_command"] % defaults))

    
env["BUILDERS"]["ReleaseJobs"] = Builder(action="qrls -u ${USER}")


env['PRINT_CMD_LINE_FUNC'] = print_cmd_line
env.Decider("timestamp-newer")


for task in env["TASKS"]:
    classified_items = []
    task_name = task["name"]
    train_proportion = task.get("train_proportion", defaults.get("train_proportion", .9))
    input_file = env.File(task["file"])
    count_file = env.GetCount("work/%s_total.txt.gz" % task_name, input_file)
    for size in task.get("sizes", defaults.get("sizes", [])):
        for fold in range(1, task.get("folds", defaults.get("folds", 1)) + 1):
            train, test = env.CreateSplit(["work/%s_%s_%s_%s.txt.gz" % (task_name, x, size, fold) for x in ["train", "test"]],
                                               [count_file, env.Value(size), env.Value(train_proportion)])
            for model in env["MODELS"]:
                model_name = model["name"]
                train_builder = env["BUILDERS"]["Train %s" % model_name]
                apply_builder = env["BUILDERS"]["Apply %s" % model_name]
                
                model = train_builder(env,
                                      "work/%s_%s_%s_%s.model" % (task_name, model_name, size, fold),
                                      [train, input_file])

                #classified = [env.File("work/%s_%s_%s_%s_results.txt.gz" % (task_name, model_name, size, fold))]
                classified = apply_builder(env,
                                           "work/%s_%s_%s_%s_results.txt.gz" % (task_name, model_name, size, fold),
                                           [model, test, input_file])
                
                classified[0].Tag("size", size)
                classified[0].Tag("task", task_name)
                classified[0].Tag("fold", fold)
                classified[0].Tag("model", model_name)
                classified_items.append(classified)

    if len(classified_items) > 0:
    #    compare = env.Compare("work/%s_compare.txt.gz" % (task_name), classified_items)
        scores = env.Evaluate("work/%s_scores.txt.gz" % (task_name), classified_items)
        plots = env.Plot("work/%s_plot.png" % (task_name), scores)
