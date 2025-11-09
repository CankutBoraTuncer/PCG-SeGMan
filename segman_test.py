import robotic as ry
import sys
sys.path.append('..')
from MOSeGMan.MOSeGMan import MOSeGMan
import os
import random
import math

random.seed(44)
verbose = 0

ry.params_clear()
ry.params_add({"Render/lights":[-5.,0.,5., -5.,0.,5.]})
base_path = os.path.abspath(os.path.dirname(__file__))


if __name__ == "__main__":
    folder = "ry_config/case_run_id_32/"
    # Run for all pcg-i folders in the case_run_id_18 folder by getting the numnber of folders
    n_folders = len([name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name)) and name.startswith("pcg-")])
    for i in range(n_folders):
        C = ry.Config()
        C_hm = ry.Config()
        C.addFile(os.path.join(folder, f"pcg-{i}/pcg-{i}.g"))
        C_hm.addFile(os.path.join(folder, f"pcg-{i}/pcg-{i}-aux.g"))
        segman = MOSeGMan(C, C_hm, verbose=verbose)
        if segman.run():
            segman.display_solution(pause = 0.01)
            






