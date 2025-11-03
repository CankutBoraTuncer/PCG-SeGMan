from stable_baselines3.common.env_checker import check_env
from grid_pcg_env import GridPCGEnv

env = GridPCGEnv(size=13, max_steps=96, seed=0)
check_env(env)
print("Env check passed ✅")

