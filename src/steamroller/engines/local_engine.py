from SCons.Builder import Builder
from .grid_engine import GridEngine


class LocalEngine(GridEngine):

    name = "local"
    parameters = {}
    
    @property
    def queues(self):
        return []

    @classmethod
    def available(cls, *argv, **argd) -> bool:
        return True

    def create_builder(self, env, builder):
        return builder
