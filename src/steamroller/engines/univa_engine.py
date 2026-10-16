import os
import subprocess
import logging
import shlex
from .grid_engine import GridEngine

#def univa(commands, name, std, dep_ids=[], grid_resources=[], working_dir=None, queue="all.q"):
def univa(commands, name, std, dep_ids=[], working_dir=None, gpu_count=0, time="48:00:00", memory="8G", queue=None, account=None):
    if not isinstance(commands, list):
        commands = [commands]
    if os.path.exists(std):
        try:
            os.remove(std)
        except:
            pass
    deps = "" if len(dep_ids) == 0 else "-hold_jid {}".format(",".join([str(x) for x in dep_ids]))
    #res = "" if len(grid_resources) == 0 else "-l {}".format(",".join([str(x) for x in grid_resources]))
    wd = "-wd {}".format(working_dir) if working_dir else "-cwd"
    qcommand = "qsub -terse -shell n -V -N {} -q {} -b n {} {} -j y -o {} -l h_rt={},mem_free={}".format(name, queue, wd, deps, std, time, memory)
    logging.info("\n".join(commands))
    p = subprocess.Popen(shlex.split(qcommand), stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    out, err = p.communicate("\n".join(commands).encode())
    return int(out.strip())


class UnivaEngine(GridEngine):

    @property
    def queues(self):
        return []

    @classmethod
    def available(cls, *argv, **argd) -> bool:
        if cls.check_for_executable("qsub"):
            pid = subprocess.Popen(["qsub", "-help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = pid.communicate()
        return False

