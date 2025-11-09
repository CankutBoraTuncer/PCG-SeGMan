import os
from contextlib import contextmanager
import robotic as ry
import random 

# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #

class Node():
    def __init__(self, C:ry.Config, C_hm:ry.Config, objs:list, obj_weights:dict, obj_scores:dict, obj_pot_raw_score:float=0, layer:int=0, FS:list=[], reachable_objs:set=set(), moved_obj:str=None, is_first:bool=False, scene_score_raw_prev:float=0, reloc_o:set = set()):
        self.C = ry.Config()
        self.C.addConfigurationCopy(C)
        self.C_hm = ry.Config()
        self.C_hm.addConfigurationCopy(C_hm) 
        self.objs = objs
        self.obj_weights = obj_weights
        self.obj_scores = obj_scores
        self.obj_pot_raw_score = obj_pot_raw_score
        self.layer = layer
        self.visit = 1
        self.FS = FS
        self.total_score = float("-inf")
        self.scene_score = 0
        self.scene_score_raw = 0
        self.scene_score_raw_prev = scene_score_raw_prev 
        self.id = random.randint(0, 1000000)
        self.pts = {}
        self.is_first = is_first
        self.pp_seq = 0
        self.reachable_objs = reachable_objs
        self.moved_obj = moved_obj
        self.reloc_o = reloc_o

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.total_score == other.total_score
    
    def __lt__(self, other):
        if not isinstance(other, Node):
            return False
        return self.total_score > other.total_score
    
    def __str__(self):
        return "Node: layer={}, visit={}, pairs={}, total_score={}, id={}".format(self.layer, self.visit, self.objs, self.total_score, self.id)
    
    def __repr__(self):
        return self.__str__()

# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #

@contextmanager
def suppress_stdout():
    devnull = os.open(os.devnull, os.O_WRONLY)
    original_stdout = os.dup(1)
    original_stderr = os.dup(2)
    
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(original_stdout, 1)
        os.dup2(original_stderr, 2)
        os.close(devnull)
        os.close(original_stdout)
        os.close(original_stderr)
