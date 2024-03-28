from .grid_engine import GridEngine
from .univa_engine import UnivaEngine
from .slurm_engine import SlurmEngine
from .local_engine import LocalEngine

registry = {}

def register_engine(engine_name, engine_class):
    if issubclass(engine_class, GridEngine):
        registry[engine_name] = engine_class
    else:
        raise Exception("The class '{}' is not a subclass of GridEngine and so cannot be registered!".format(str(engine_class)))

register_engine("univa", UnivaEngine)
register_engine("slurm", SlurmEngine)
register_engine("local", LocalEngine)
