import os
import os.path
import logging
import random
import subprocess
import shlex
from data_io import reader, writer
import gzip
import re
import pickle


vars = Variables("custom.py")
vars.AddVariables(
    ("OUTPUT_WIDTH", "Upper limit on how long a debugging line will be before it's truncated", 1000),
    ("DEFAULTS", "General variables (potentially overridden by models and tasks)", {}),
    ("MODELS", "Classification models to compare", []),
    ("TASKS", "Classification tasks", []),
    ("TEST_COUNT", "Data size for testing models", 10000),
    BoolVariable("GRID", "Do we have access to a grid via the qsub command?", False),
    ("GRID_RESOURCES", "List of resources to request for a job", []),
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


def resource_emitter(target, source, env):
    return (target + [target[0].rstr() + ".resources.txt"], source)    


def qsub(command, dep_ids=[], grid_resources=[]):
    deps = "" if len(dep_ids) == 0 else "-hold_jid %s" % (",".join([str(x) for x in dep_ids]))
    res = "" if len(grid_resources) == 0 else "-l %s" % (",".join([str(x) for x in grid_resources]))
    qcommand = "qsub -v PATH -v PYTHONPATH -b y -cwd -j y -terse -o ${TARGETS[-1]} %s %s %s" % (deps, res, command)
    p = subprocess.Popen(shlex.split(qcommand), stdout=subprocess.PIPE)
    out, err = p.communicate()
    return int(out.strip())


#def localsub(command):
#    qcommand = "qsub -v PATH -v PYTHONPATH -b y -cwd -j y -terse -o output.out %s %s %s" % (deps, res, command)
#    p = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
#    out, err = p.communicate()
#    return int(out.strip())


# def Grid(command, grid_resources=[]):
def grid_method(target, source, env):
    depends_on = set(filter(lambda x : x != None, [s.GetTag("built_by_job") for s in source]))
    job_id = qsub(env.subst(command, source=source, target=target), depends_on, env["GRID_RESOURCES"])
    for t in target:
        t.Tag("built_by_job", job_id)
    logging.info("Job %d depends on %s", job_id, depends_on)
    return None
#     return Builder(action=Action(grid_method, "grid(%s)" % (command)))




# def Local(command):
#     def local_emitter(target, source, env):
#         return target + [target[0].rstr() + ".resources.gz"], source
#     def local_method(target, source, env):
#         #local_command = "/usr/bin/time --verbose " + command
#         p = subprocess.Popen(shlex.split(env.subst(local_command, source=source, target=target)),
#                              stdout=subprocess.PIPE,
#                              stderr=subprocess.PIPE)
#         out, err = p.communicate()
#         with writer(gzip.open(target[-1].rstr(), "w")) as ofd:
#             ofd.write(err + out)
#         return None
#     return Builder(action=Action(local_method, "%s > ${TARGETS[-1]}" % (command)), emitter=local_emitter)


def Wrapper(command, grid_resources=[]):
    timed_command = "/usr/bin/time --verbose -o ${TARGETS[-1]} " + command
    if env["GRID"] and isinstance(local_command, basestring):
        return Grid(timed_command, grid_resources)
    else:
        return Builder(action=timed_command, emitter=resource_emitter)


defaults = env["DEFAULTS"]


for name, command in [
        ("GetCount", "python tools/get_count.py --input ${SOURCES[0]} --output ${TARGETS[0]}"),
        ("CreateSplit", "python tools/create_split.py --total_file ${SOURCES[0]} --train_count ${SOURCES[1].read()} --test_count ${TEST_COUNT} --train ${TARGETS[0]} --test ${TARGETS[1]}"),
        ("Evaluate", "python tools/evaluate.py -o ${TARGETS[0]} ${SOURCES}"),
        ("CollateResources", "python tools/collate_resources.py -o ${TARGETS[0]} ${SOURCES}"),
        ("ModelSizes", "python tools/model_sizes.py -o ${TARGETS[0]} ${SOURCES}"),        
        ("Plot", "python tools/plot.py -o ${TARGETS[0]} -f ${FIELD} ${SOURCES}"),
]:
    env["BUILDERS"][name] = Wrapper(command % defaults)


for model in env["MODELS"]:
    env["BUILDERS"]["Train %s" % model["name"]] = Wrapper(model["train_command"] % defaults)
    env["BUILDERS"]["Apply %s" % model["name"]] = Wrapper(model["apply_command"] % defaults)


env['PRINT_CMD_LINE_FUNC'] = print_cmd_line
env.Decider("timestamp-newer")


for task in env["TASKS"]:
    classified_items = []
    train_resource_list = []
    apply_resource_list = []
    model_list = []
    task_name = task["name"]
    train_proportion = task.get("train_proportion", defaults.get("train_proportion", .9))
    input_file = env.File(task["file"])
    count_file, resources = env.GetCount("work/%s_total.txt.gz" % task_name, input_file)
    for size in task.get("sizes", defaults.get("sizes", [])):
        for fold in range(1, task.get("folds", defaults.get("folds", 1)) + 1):
            train, test, resources = env.CreateSplit(["work/%s_%s_%s_%s.txt.gz" % (task_name, x, size, fold) for x in ["train", "test"]],
                                               [count_file, env.Value(size), env.Value(train_proportion)])
            for model in env["MODELS"]:
                model_name = model["name"]
                train_builder = env["BUILDERS"]["Train %s" % model_name]
                apply_builder = env["BUILDERS"]["Apply %s" % model_name]
                
                model, resources = train_builder(env,
                                                 "work/%s_%s_%s_%s.model" % (task_name, model_name, size, fold),
                                                 [train, input_file])
                train_resource_list.append(resources)
                model_list.append(model)
                
                classified, resources = apply_builder(env,
                                                      "work/%s_%s_%s_%s_results.txt.gz" % (task_name, model_name, size, fold),
                                                      [model, test, input_file])
                apply_resource_list.append(resources)
                classified_items.append(classified)
                

    if len(classified_items) > 0:
        scores, _ = env.Evaluate("work/%s_scores.txt.gz" % (task_name), classified_items)
        train_resources, _ = env.CollateResources("work/%s_trainresources.txt.gz" % (task_name), train_resource_list)
        apply_resources, _ = env.CollateResources("work/%s_applyresources.txt.gz" % (task_name), apply_resource_list)
        model_sizes, _ = env.ModelSizes("work/%s_modelsizes.txt.gz" % (task_name), model_list)
        env.Plot("work/%s_trainmemory_plot.png" % (task_name), train_resources, FIELD="Memory")
        env.Plot("work/%s_traincpu_plot.png" % (task_name), train_resources, FIELD="CPU")
        env.Plot("work/%s_applymemory_plot.png" % (task_name), apply_resources, FIELD="Memory")
        env.Plot("work/%s_applycpu_plot.png" % (task_name), apply_resources, FIELD="CPU")
        env.Plot("work/%s_modelsize_plot.png" % (task_name), model_sizes, FIELD="Size")        
        env.Plot("work/%s_fscore_plot.png" % (task_name), scores, FIELD="F-Score")
