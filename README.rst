------
SteamRoller |Logo|
------

------
Quick start
------

As of version 1.3, SteamRoller is now intended to be used directly in a normal SCons build setup.  Simply use the grid-aware Builder function and any command-line rule should just work, i.e. these two builders will produce the same results::

  from steamroller import GridBuilder as Builder  
  my_builder = Builder("my_command ${SOURCES} ${TARGETS}")
  my_grid_builder = GridBuilder("my_command ${SOURCES} ${TARGETS}")
  
The command can also be a list that will be run in sequence, and when invoked can specify resources and the destination queue::

  my_grid_builder = GridBuilder(["module load cuda90/toolkit", "my_command ${SOURCES} ${TARGETS}"])
  output = my_grid_builder("work/output.txt", "work/input.txt",
                           GRID_QUEUE="gpu.q",
			   GRID_RESOURCES=["gpu=1", "h_rt=10:0:0"])

SteamRoller also provides a nice wrapper to help create command-line rules that track their own source files and arguments.  The following will create a builder that, given source `input.txt` and with the `DEPTH` variable set to `3`, executes the command `python src/my_script.py -i input.txt --depth 3`, and any change to the script file or `src/library.py` invalidates the cache::

  from steamroller import action_maker
  my_builder = Builder(**action_maker(interpreter="python",
                                      script="src/my_script.py",
                                      args="-i ${SOURCES}",
				      other_args=["DEPTH"],
				      other_deps=["src/library.py"]))

You can also put any other command-line switches, custom variable-substitutions, etc in the `args` parameter and it will be processed accordingly.

One minor difference between a regular builder and its grid-aware counterpart is the latter will redirect the grid stdout/stderr to a file based on the name of the first target ("${TARGETS[0]}.qout", so for the above example, "work/output.txt.qout").

------
Getting started with an example
------

Here's a reasonable way to structure a new experiment.  Assuming you're starting in an empty directory named after the experiment (e.g. `~/my_experiment`) and have a recent version of Python 3 on your path::

  python -m venv ~/venvs/my_experiment
  source ~/venvs/my_experiment/bin/activate
  pip install scons steamroller
  mkdir data src work
  
The idea is that untouched data goes under `data/`, scripts and such for steps in your pipeline go in `src/`, and all outputs, tracked by the build system, will go under `work/`.  Create a file called `SConstruct` with this content::

  import os
  from steamroller import action_maker

  vars = Variables("custom.py")
  vars.AddVariables(
    BoolVariable("USE_GRID", "Run via qsub rather than locally", False),
    BoolVariable("USE_GPU", "Run on GPU", False),
    ("GRID_CPU_QUEUE", "", "all.q"),
    ("GRID_CPU_RESOURCES", "", ["h_rt=100:0:0"]),
    ("GRID_GPU_QUEUE", "", "gpu.q@@1080"),
    ("GRID_GPU_RESOURCES", "", ["h_rt=100:0:0", "gpu=1"]),
    ("GPU_PREAMBLE", "", "module load cuda90/toolkit"),    
  )

  env = Environment(variables=vars, ENV=os.environ)

  gpu_preamble = [env["GPU_PREAMBLE"] if env["USE_GPU"] else []]
  model_queue = env["GRID_GPU_QUEUE"] if env["USE_GPU"] else env["GRID_CPU_QUEUE"]
  model_resources = env["GRID_GPU_RESOURCES"] if env["USE_GPU"] else env["GRID_CPU_RESOURCES"]
  if env["USE_GRID"]:
    from steamroller import GridBuilder as Builder

This boilerplate does a number of things, but most importantly, it ensures that build rules defined afterwards are grid-aware, if requested.  At this point you have defined no build rules or targets, so invoking the system does nothing::

  (my_experiment) [tlippincott@test4 my_experiment]$ scons
  scons: Reading SConscript files ...
  scons: done reading SConscript files.
  scons: Building targets ...
  scons: `.' is up to date.
  scons: done building targets.

Next, try adding a couple build rules, using the `action_maker` helper function::

  env.Append(BUILDERS={"Split" : Builder(**action_maker()),
                       "Train" : Builder(**action_maker()),
		       "Apply" : Builder(**action_maker()),
		       "Plot" : Builder(**action_maker())
		       })

Finally, describe how to run your experiment, in terms of build rules and the targets they produce::

  data = env.File("data/my_data.txt.gz")
  train_data, test_data = env.Split(["work/train.txt.gz", "work/test.txt.gz"], data)
  results = []
  for param in range(100):
    model = env.Train("work/model_${PARAM}.gz", train_data, PARAM=param)
    results.append(env.Apply("work/result_${PARAM}.gz", [model, test_data], PARAM=param))
  figure = env.Plot("work/figure.png", results)



----
FAQ
----

.. |Logo|   image:: logo.png
