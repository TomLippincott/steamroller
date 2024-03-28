from abc import ABC, abstractmethod
import typing
import os
import os.path


class GridEngine(ABC):

    @classmethod
    def check_for_executable(cls, name: str) -> bool:
        return any([os.path.exists(os.path.join(p, name)) for p in os.environ["PATH"].split(":")])
    
    @property
    @abstractmethod
    def queues(self):
        raise NotImplemented()

    @classmethod
    @abstractmethod
    def available(cls, *argv, **argd) -> bool:
        raise NotImplemented()

    #@abstractmethod
    def submit(
            commands,
            name,
            std,
            dep_ids=[],
            working_dir=None,
            gpu_count=0,
            time="12:00:00",
            memory="8G",
            queue=None,
            account=None
    ):
        pass
