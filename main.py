from reader_functions import *

simdata_path = "./simdata2021"

maneuvers = get_maneuvers(simdata_path)

for maneuver in maneuvers:
    plot_path_and_speeds(f"{simdata_path}/{maneuver}")
