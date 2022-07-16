------
SteamRoller |Logo|
------

------
Getting started with an example
------

First and foremost, steamroller builds upon the [SCons](https://scons.org/), which has a [very in-depth MAN page](https://scons.org/doc/production/HTML/scons-man.html) and [API](https://scons.org/doc/latest/HTML/scons-api/index.html), but a probably-more-useful [user guide](https://scons.org/doc/production/HTML/scons-user/index.html).

Here's a reasonable way to structure a new experiment.  Assuming you're starting in an empty directory named after the experiment (e.g. `~/my_experiment`) and have a recent version of Python 3 on your path::

  $ python -m venv ~/venvs/my_experiment
  $ source ~/venvs/my_experiment/bin/activate
  (my_experiment) $ pip install pip -U
  (my_experiment) $ pip install steamroller
  (my_experiment) $ mkdir data src work
  
The idea is that untouched data goes under `data/`, scripts and such for steps in your pipeline go in `src/`, and all outputs, tracked by the build system, will go under `work/`.  Create a file called `SConstruct` with this content::

  import os
  import steamroller

  env = Environment(variables=vars, ENV=os.environ, tools=[steamroller.generate])
  env.Decider("timestamp-newer")

This boilerplate does a number of things, but most importantly, it ensures that build rules defined afterwards are grid-aware, if requested.  At this point you have defined no build rules or targets, so invoking the system does nothing::

  (my_experiment) $ scons
  scons: Reading SConscript files ...
  scons: done reading SConscript files.
  scons: Building targets ...
  scons: `.' is up to date.
  scons: done building targets.

Next, try adding a few build rules as normal::

  env.Append(
      BUILDERS={
          "Split" : env.Builder(action="python src/split.py ${SOURCES[0]} ${TARGETS[0]} ${TARGETS[1]}"),
	  "Train" : env.Builder(action="python src/train.py ${SOURCES[0]} --param ${PARAM} ${TARGETS[0]}"),
	  "Apply" : env.Builder(action="python src/apply.py ${SOURCES} ${TARGETS[0]}"),
          "Plot" : env.Builder(action="python src/plot.py ${SOURCES} ${TARGETS[0]}")
      }
  )

None of these scripts exist yet, so just for the sake of this example, let's make them, and a data file::

  (my_experiment) $ touch data/my_data.txt.gz
  (my_experiment) $ touch src/{split,train,apply,plot}.py

Finally, describe how to run your experiment, in terms of build rules and the targets they produce::

  data = env.File("data/my_data.txt.gz")
  results = []
  for split in range(5):
      train, test = env.Split(["work/train_${SPLIT}.txt", "work/test_${SPLIT}.txt"], data, SPLIT=split)
      for param in range(20):
          model = env.Train("work/model_${SPLIT}_${PARAM}.bin", train, SPLIT=split, PARAM=param)
          results.append(env.Apply("work/output_${SPLIT}_${PARAM}.out", [model, test], SPLIT=split, PARAM=param))
  figure = env.Plot("work/figure.png", results)

You can now invoke locally::

  (my_experiment) $ scons -n

Or on the grid::

  (my_experiment) $ scons -n USE_GRID=True

The first invocation should print out the commands it would run, the second should print them out along with some info on how they would be submitted to the grid.

----
FAQ
----

.. |Logo|   image:: logo.png
