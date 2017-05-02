import functools
from SCons.Action import Action
from SCons.Builder import Builder
import SCons.Util
import subprocess
import logging
import time

def qsub(command, name, std, dep_ids=[], grid_resources=[]):
    deps = "" if len(dep_ids) == 0 else "-hold_jid {}".format(",".join([str(x) for x in dep_ids]))
    res = "" if len(grid_resources) == 0 else "-l {}".format(",".join([str(x) for x in grid_resources]))
    qcommand = "qsub -v PATH -v PYTHONPATH -N {} -b y -cwd -j y -terse -o {} {} {}".format(name, std, deps, res) + command
    p = subprocess.Popen(shlex.split(qcommand), stdout=subprocess.PIPE)
    out, err = p.communicate()
    return int(out.strip())





# def Wrapper(env, command, name, script, grid_resources=[]):
#     def emitter(target, source, env):
#         [env.Depends(t, script) for t in target]
#         if env["GRID"]:
#             return (target + [target[0].rstr() + ".gridout.txt", target[0].rstr() + ".resources.txt"], source)    
#         else:
#             return (target + [target[0].rstr() + ".resources.txt"], source)
#     deps = ""
#     res = ""
#     timed_command = "/usr/bin/time --verbose -o ${{TARGETS[-1]}} " + command
#     actual_command = "qsub -v PATH -v PYTHONPATH -N ${TASK_NAME} -b y -cwd -j y -terse -o ${TARGETS[-2]} ${DEPS} ${RES} ${COMM}"
#     #.format(name, deps, res, timed_command) if env["GRID"] else timed_command
#     return Builder(generator=command_generator, emitter=emitter)
#     # if env["GRID"] and isinstance(command, basestring):
#     #     timed_command = "/usr/bin/time --verbose " + command
#     #     deps = "" if len(dep_ids) == 0 else "-hold_jid %s" % (",".join([str(x) for x in dep_ids]))
#     #     res = "" if len(grid_resources) == 0 else "-l %s" % (",".join([str(x) for x in grid_resources]))
#     #     qcommand = "qsub -v PATH -v PYTHONPATH -N %s -b y -cwd -j y -terse -o %s %s %s %s" % (name, std, deps, res, command)
#     #     return Builder(action=Action(functools.partial(grid_method, timed_command, name), 
#     #                                  "grid(%s -> ${TARGETS[0]})" % (command if env["VERBOSE"] else name)), 
#     #                    emitter=resource_emitter)
#     # else:
#     #     timed_command = "/usr/bin/time --verbose -o ${TARGETS[-1]} " + command
#     #     return Builder(action=Action(timed_command, "local({})".format(command)),
                                     
#     #                                  #"local(%s -> ${TARGETS[0]})" % (command if env["VERBOSE"] else name)), 
#     #                    emitter=resource_emitter)

# def make_action_generator(base_command):
# #    timed_command = "/usr/bin/time --verbose -o ${TARGETS[-1]} " + base_command
# def command_generator(target, source, env, for_signature):
#     if env["GRID"]:
#             deps = set(filter(lambda x : x != None, [s.GetTag("built_by_job") for s in source]))
#             dep_string = "" if len(deps) == 0 else "-hold_jid {}".format(",".join([str(d) for d in deps]))
#             #[t for t in [str(s.GetTag("built_by_job")) for s in source] if t]))
#             res_string = "" if len(env["GRID_RESOURCES"]) == 0 else "-l %s" % (",".join([str(x) for x in env["GRID_RESOURCES"]]))
#             sub = "$(qsub -v PATH -v PYTHONPATH -N ${TASK_NAME} -b y -cwd -j y -terse -o ${TARGETS[-2]} " + dep_string + " " + res_string + " $)"
#             #print sub
#             command = sub + timed_command
#         else:
#             command = timed_command
#         return Action(command)
#     return command_generator


def make_emitter(script):
    def emitter(target, source, env):
        [env.Depends(t, script) for t in target]
        return (target + [target[0].rstr() + ".resources.txt"], source)
    return emitter


def make_builder(env, base_command):
    timed_command = "/usr/bin/time --verbose -o ${TARGETS[-1]} " + base_command
    def grid_method(target, source, env):
        depends_on = set(filter(lambda x : x != None, [s.GetTag("built_by_job") for s in source]))
        job_id = qsub(env.subst(timed_command, source=source, target=target), base_command, "{}.qout".format(target[-1].rstr()), depends_on, env["GRID_RESOURCES"])
        for t in target:
            t.Tag("built_by_job", job_id)
        logging.info("Job %d depends on %s", job_id, depends_on)
        return None

    return Builder(action=grid_method if env["GRID"] else timed_command, emitter=make_emitter(base_command.split()[0]))


def generate(env):

    env.SetDefault(
        MAX_NGRAM=4,
    )

    for name, script in [
            ("GetCount", "site_scons/get_count.py --input ${SOURCES[0]} --output ${TARGETS[0]}"),
            ("CreateSplit", "site_scons/create_split.py --total_file ${SOURCES[0]} --train_count ${TRAIN_COUNT} --test_count ${TEST_COUNT} --train ${TARGETS[0]} --test ${TARGETS[1]}"),
            ("Evaluate", "site_scons/evaluate.py -o ${TARGETS[0]} ${SOURCES}"),
            ("CollateResources", "site_scons/collate_resources.py -o ${TARGETS[0]} ${SOURCES}"),
            ("ModelSizes", "site_scons/model_sizes.py -o ${TARGETS[0]} ${SOURCES}"),        
            ("Plot", "site_scons/plot.py -o ${TARGETS[0]} -f \"${FIELD}\" ${SOURCES}"),
    ]:
        env["BUILDERS"][name] = make_builder(env, script)
        #Builder(generator=make_action_generator(script),
        #                                emitter=make_emitter(script.split()[0]))
        
    for model in env["MODELS"]:
        env["BUILDERS"]["Train{}".format(model["name"])] = make_builder(env, model["train_command"])
#Builder(generator=make_command_generator(model["train_command"]), 
#                                                                   emitter=make_emitter(model["train_command"].split()[0]))
        env["BUILDERS"]["Apply{}".format(model["name"])] = make_builder(env, model["apply_command"])
        #Builder(generator=make_command_generator(model["apply_command"]), 
        #                                                           emitter=make_emitter(model["apply_command"].split()[0]))

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
