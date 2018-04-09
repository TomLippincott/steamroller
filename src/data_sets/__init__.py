import os.path
import re

def split_generator(target, source, env, for_signature):
    sizes = env.get("SIZES", None)
    if not sizes:
        spec = "--train_size 0.0 --dev_size 0.0 --test_size 0.0"
    else:
        spec = ""
    return "python -m steamroller.data_sets.split {} --train_output ${{TARGETS[0]}} --dev_output ${{TARGETS[1]}} --test_output ${{TARGETS[2]}} ${{SOURCES}}".format(spec)

def split_emitter(target, source, env):
    new_target = [os.path.join("work", "${{NAME}}_{}.idx.gz".format(n)) for n in ["train", "dev", "test"]]
    return (new_target, source)

BUILDERS = {
    "Split" : (split_generator, split_emitter),
}
