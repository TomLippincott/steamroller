import subprocess
import json
import re
from .grid_engine import GridEngine


class SlurmEngine(GridEngine):

    name = "slurm"

    def __init__(self):
        pass
        #pid = subprocess.Popen(["sacct", "-lL", "--json"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #stdout, stderr = pid.communicate()
        #self.state = json.loads(stdout.decode("utf-8"))
        #pid = subprocess.Popen(["sinfo", "--json"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #stdout, stderr = pid.communicate()
        #self.info = json.loads(stdout.decode("utf-8"))

    def job_names(self):
        return set([j["name"] for j in self.state["jobs"]])
    
    @classmethod
    def available(cls, *argv, **argd) -> bool:
        return cls.check_for_executable("sacct")

    submit_string = " ".join(
        [
            "sbatch",
            "-J ${STEAMROLLER_NAME}",
            "--kill-on-invalid-dep=yes",
            "--mail-type=NONE",
            "-o ${STEAMROLLER_LOG}",
            "--parsable",
            "${'-D ' + STEAMROLLER_WORKING_DIRECTORY if STEAMROLLER_WORKING_DIRECTORY else ''}",
            "${'-A ' + STEAMROLLER_ACCOUNT if STEAMROLLER_ACCOUNT else ''}",
            "${'-t ' + STEAMROLLER_TIME if STEAMROLLER_TIME else ''}",
            "${'--mem=' + STEAMROLLER_MEMORY if STEAMROLLER_MEMORY else ''}",
            "${'-p ' + STEAMROLLER_QUEUE if STEAMROLLER_QUEUE else ''}",
            "${'--gres=gpu:' + str(STEAMROLLER_GPU_COUNT) if STEAMROLLER_GPU_COUNT else ''}",
            "${'-d afterok:' + ':'.join(map(str, STEAMROLLER_DEPENDENCIES)) if STEAMROLLER_DEPENDENCIES else ''}",            
        ]
    )
