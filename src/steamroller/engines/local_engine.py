from .grid_engine import GridEngine

class LocalEngine(GridEngine):

    @property
    def queues(self):
        return []

    @classmethod
    def available(cls, *argv, **argd) -> bool:
        return True

    def create_builder(self, env, **args):
        return Builder(**args)
