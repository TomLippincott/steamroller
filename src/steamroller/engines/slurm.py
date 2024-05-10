import subprocess
import re
from .grid_engine import GridEngine
from ..util import GridAwareBuilder, ActionMaker

def slurm(commands, name, std, dep_ids=[], working_dir=None, gpu_count=0, time="12:00:00", memory="8G", queue=None, account=None):
    if not isinstance(commands, list):
        commands = [commands]
    if os.path.exists(std):
        try:
            os.remove(std)
        except:
            pass
    deps = "" if len(dep_ids) == 0 else "-d afterok:{}".format(":".join([str(x) for x in dep_ids]))
    wd = "-D {}".format(working_dir) if working_dir else "" #"-cwd"
    acct = "-A {}".format(account) if account else "" #"-cwd"
    queue = "-p {}".format(queue) if queue else "" #"-cwd"
    gpus = "--gres=gpu:{}".format(gpu_count) if gpu_count else ""
    qcommand = "sbatch {wd} {deps} -J {name} --kill-on-invalid-dep=yes --mail-type=NONE --mem={memory} -o {std} --parsable -t {time} {acct} {queue} {gpus}".format(
        name=name,
        deps=deps,
        wd=wd,
        std=std,
        time=time,
        memory=memory,
        acct=acct,
        queue=queue,
        gpus=gpus
    )
    logging.info("\n".join(commands))
    commands = ["#!/bin/bash"] + commands
    p = subprocess.Popen(shlex.split(qcommand), stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    out, err = p.communicate("\n".join(commands).encode())
    return int(out.strip())


class SlurmEngine(GridEngine):

    @property
    def queues(self):
        return []    
    
    @classmethod
    def available(cls, *argv, **argd) -> bool:
        if cls.check_for_executable("sinfo"):
            pid = subprocess.Popen(["sinfo", "-V"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = pid.communicate()
            if re.match(r"slurm (.*)\n", stdout.decode("utf-8")):
                return True
        return False

    def create_builder(self, env, *argv, **argd):
        return GridAwareBuilder(
            env,
            **argd
        )
