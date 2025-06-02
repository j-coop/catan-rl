from envs.base_env.env import CatanBaseEnv
from visualization.map_plotter import CatanMapPlotter

env = CatanBaseEnv()
obs = env.reset()

plotter = CatanMapPlotter(obs)
plotter.plot_catan_map()
