sconstruct = """
import os
import os.path
import logging
import random
import subprocess
import shlex
import gzip
import re
import functools
import time
import imp
import sys
del sys.modules['pickle']

sys.modules['pickle'] = imp.load_module('pickle', *imp.find_module('pickle'))

import pickle
import steamroller.scons


vars = Variables("steamroller_config.py")
vars.AddVariables(
    ("OUTPUT_WIDTH", "Upper limit on how long a debugging line will be before it's truncated", 1000),
    ("VERBOSE", "Whether to print the full commands being executed", False),
    ("DEFAULTS", "General variables (potentially overridden by models and tasks)", {}),
    ("MODELS", "Classification models to compare", []),
    ("TASKS", "Classification tasks", []),
    ("TEST_COUNT", "Data size for testing models", 10000),
    BoolVariable("GRID", "Do we have access to a grid via the qsub command?", False),
    ("GRID_RESOURCES", "List of resources to request for a job", []),
    ("GRID_CHECK_INTERVAL", "How many seconds between checking on job status", 30),
)

def print_cmd_line(s, target, source, env):
    if len(s) > int(env["OUTPUT_WIDTH"]):
        print s[:int(env["OUTPUT_WIDTH"]) - 10] + "..." + s[-7:]
    else:
        print s


env = Environment(variables=vars, ENV=os.environ, TARFLAGS="-c -z", TARSUFFIX=".tgz",
                  tools=["default", steamroller.scons.generate],
)


logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)


defaults = env["DEFAULTS"]


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
    count_file, _ = env.GetCount("work/${TASK_NAME}_total.txt.gz", input_file, TASK_NAME=task_name)

    
    for train_count in task.get("sizes", defaults.get("sizes", [])):
        for fold in range(1, task.get("folds", defaults.get("folds", 1)) + 1):

            train, test, _ = env.CreateSplit(["work/${TASK_NAME}_train_${FOLD}_${TRAIN_COUNT}_${TEST_COUNT}.txt.gz",
                                           "work/${TASK_NAME}_test_${FOLD}_${TRAIN_COUNT}_${TEST_COUNT}.txt.gz"],
                                          count_file, FOLD=fold, TRAIN_COUNT=train_count, TASK_NAME=task_name)

            for model in env["MODELS"]:
                model_name = model["name"]
                train_builder = env["BUILDERS"]["Train%s" % model_name]
                apply_builder = env["BUILDERS"]["Apply%s" % model_name]
                model_file, resources = train_builder(env,
                                                      "work/${TASK_NAME}_${MODEL_NAME}_${TRAIN_COUNT}_${FOLD}.model.gz",
                                                      [train, input_file],
                                                      FOLD=fold, TRAIN_COUNT=train_count, TASK_NAME=task_name, MODEL_NAME=model_name,
                                                      GRID_RESOURCES=model.get("grid_resources", env["GRID_RESOURCES"]),
                                                  )


                train_resource_list.append(resources)
                model_list.append(model_file)
                classified, _ = apply_builder(env,
                                              "work/${TASK_NAME}_${MODEL_NAME}_${TRAIN_COUNT}_${FOLD}_probabilities.txt.gz",
                                              [model_file, test, input_file],
                                              FOLD=fold, TRAIN_COUNT=train_count, TASK_NAME=task_name, MODEL_NAME=model_name,
                                              GRID_RESOURCES=model.get("grid_resources", []),
                )
                apply_resource_list.append(resources)
                classified_items.append(classified)
                continue
    plots = []
    if len(classified_items) > 0:
        scores, _ = env.Evaluate("work/%s_scores.txt.gz" % (task_name), classified_items)
        train_resources, _ = env.CollateResources("work/%s_trainresources.txt.gz" % (task_name), train_resource_list)
        apply_resources, _ = env.CollateResources("work/%s_applyresources.txt.gz" % (task_name), apply_resource_list)
        model_sizes, _ = env.ModelSizes("work/%s_modelsizes.txt.gz" % (task_name), model_list)
        env.Plot("work/%s_trainmemory_plot.png" % (task_name), train_resources, FIELD="Memory", TITLE="Max memory (G) training")
        env.Plot("work/%s_traincpu_plot.png" % (task_name), train_resources, FIELD="CPU", TITLE="CPU Time (s) training")
        env.Plot("work/%s_applymemory_plot.png" % (task_name), apply_resources, FIELD="Memory", TITLE="Max memory (G) applied to 10k instances")
        env.Plot("work/%s_applycpu_plot.png" % (task_name), apply_resources, FIELD="CPU", TITLE="CPU time (s) applied to 10k instances")
        env.Plot("work/%s_modelsize_plot.png" % (task_name), model_sizes, FIELD="Gigabytes", TITLE="Model size (G)")
        env.Plot("work/%s_fscore_plot.png" % (task_name), scores, FIELD="F_Score", TITLE="F-Score")
"""

steamroller_config = """
# These are general values that can be overridden by specific
# tasks or models.  For example, you might have a task with a
# smaller data set where `sizes` ranges too high, so you could
# override it in that task definition.
DEFAULTS = {
    # What training sizes (number of documents) to run experiments with.
    "sizes" : [1000, 2000, 4000], #8000, 16000, 32000, 64000],

    # Proportions for random data splits
    "train" : 0.9,
    "test" : 0.1,
    
    # How many random data folds for each experiment (combination
    # of size and task).  The different models are run on the same
    # folds, i.e. "task X, fold Y" is the same for models A and B.
    "folds" : 3,
}

# A "task" is just a labeled data set.
TASKS = [
    #{"name" : "LID",
    #"file" : "tasks/lid.tgz",     
    #},    
    {"name" : "Politics",
    "file" : "tasks/politics.tgz",     
    },    
    #{"name" : "Gender",
    #"file" : "tasks/gender.tgz",     
    #},    
    #{"name" : "Events",
    #"file" : "tasks/events.tgz",     
    #},
    #{"name" : "Age",
    # "file" : "tasks/age.tgz",     
    #},        
]

# A "model" provides two commands: one for training, the other for
# applying.  The entries here specify how these commands are actually
# invoked, with a few placeholders ("${SOURCES[X]}", "${TARGETS[X]}")
# for the input provided by the build system.  The commands can have
# additional placeholders, like "%(max_ngram)s", which then must be
# defined under DEFAULTS (note the additional placeholders use Python
# string substitution syntax).
MODELS = [
    
    {"name" : "NaiveBayes",
     # ${SOURCES[0]} and ${SOURCES[1]}, are the training indices and
     # input file, respectively, and ${TARGETS[0]} is the file where
     # the model will be written.
     "train_command" : "python -m steamroller.models.scikit_learn --type naive_bayes --train ${SOURCES[0]} --input ${SOURCES[1]} --output ${TARGETS[0]} --max_ngram ${MAX_NGRAM}",
     # ${SOURCES[0]}, ${SOURCES[1]} and ${SOURCES[2]} are the model file,
     # and input file, respectively, and ${TARGETS[0]} is the file to
     # write the prediction probabilities to.
     "apply_command" : "python -m steamroller.models.scikit_learn --type naive_bayes --model ${SOURCES[0]} --test ${SOURCES[1]} --input ${SOURCES[2]} --output ${TARGETS[0]}",
    },

    {"name" : "SVM",
     "train_command" : "python -m steamroller.models.scikit_learn --type svm --train ${SOURCES[0]} --input ${SOURCES[1]} --output ${TARGETS[0]} --max_ngram ${MAX_NGRAM}",
     "apply_command" : "python -m steamroller.models.scikit_learn --type svm --model ${SOURCES[0]} --test ${SOURCES[1]} --input ${SOURCES[2]} --output ${TARGETS[0]}",     
    },
    
    #{"name" : "VaLID",
    # "train_command" : "python models/valid_wrapper.py --train ${SOURCES[0]} --input ${SOURCES[1]} --output ${TARGETS[0]} --max_ngram %(max_ngram)s",
    # "apply_command" : "python models/valid_wrapper.py --model ${SOURCES[0]} --test ${SOURCES[1]} --input ${SOURCES[2]} --output ${TARGETS[0]}",
    #},

    {"name" : "LogisticRegression",
     "train_command" : "python -m steamroller.models.scikit_learn --type logistic_regression --train ${SOURCES[0]} --input ${SOURCES[1]} --output ${TARGETS[0]} --max_ngram ${MAX_NGRAM}",
     "apply_command" : "python -m steamroller.models.scikit_learn --type logistic_regression --model ${SOURCES[0]} --test ${SOURCES[1]} --input ${SOURCES[2]} --output ${TARGETS[0]}",
    },

    # {"name" : "FastText",
    #  "train_command" : "python models/fasttext_wrapper.py --train ${SOURCES[0]} --input ${SOURCES[1]} --output ${TARGETS[0]} --word_vector_size 5 --max_char_ngram %(max_ngram)s --max_word_ngram 2 --word_context_size 2",
    #  "apply_command" : "python models/fasttext_wrapper.py --model ${SOURCES[0]} --test ${SOURCES[1]} --input ${SOURCES[2]} --output ${TARGETS[0]}",
    # },

    # #{"name" : "Seal",
    # # "train_command" : "python models/seal_wrapper.py --train ${SOURCES[0]} --input ${SOURCES[1]} --output ${TARGETS[0]} --epochs 100",
    # # "apply_command" : "python models/seal_wrapper.py --model ${SOURCES[0]} --test ${SOURCES[1]} --input ${SOURCES[2]} --output ${TARGETS[0]}",
    # #},

    {"name" : "Prior",
     "train_command" : "python -m steamroller.models.scikit_learn --type prior --train ${SOURCES[0]} --input ${SOURCES[1]} --output ${TARGETS[0]} --max_ngram ${MAX_NGRAM}",
     "apply_command" : "python -m steamroller.models.scikit_learn --type prior --model ${SOURCES[0]} --test ${SOURCES[1]} --input ${SOURCES[2]} --output ${TARGETS[0]}",
    },
]

# Whether to distribute processing across a PBS system via 'qsub' et al.
GRID = False
"""

if __name__ == "__main__":

    import flask
    import argparse
    from glob import glob
    import os.path
    import logging
    
    parser = argparse.ArgumentParser("steamroller")
    parser.add_argument(dest="mode", choices=["init", "run", "serve"])
    parser.add_argument("-p", "--port", dest="port", default=8080, type=int)
    parser.add_argument("-H", "--host", dest="host", default="localhost")
    options = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    if options.mode == "init":
        if os.path.exists("SConstruct") or os.path.exists("steamroller_config.py"):
            logging.error("Refusing to overwrite existing SConstruct or steamroller_config.py files")
        else:
            with open("SConstruct", "w") as ofd:
                ofd.write(sconstruct)
            with open("steamroller_config.py", "w") as ofd:
                ofd.write(steamroller_config)     
    elif options.mode == "run":
        pass
    elif options.mode == "serve":
        
        app = flask.Flask("SteamRoller")
        images = glob("work/*png")
        
        @app.route("/")
        def browse():
            return "SteamRoller results browser"

        @app.route("/experiments/<task>")
        def experiment(task):
            return task
        
        app.run(port=options.port, host=options.host)
