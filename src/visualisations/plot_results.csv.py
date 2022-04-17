import pandas as pd
import matplotlib.pyplot as plt

mus = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

columns = ["env_name", "mu", "dist_measure_name", "vases_smashed", "spec_score", "doors_left_open"]
dist_measures = ["vase_door", "simple", "rgb"]
data = pd.read_csv("results_all.csv").filter(columns)

fig, axes = plt.subplots(3, 3, sharex=True, sharey=True)

for y, env_name in enumerate(["MuseumRush", "EasyDoorGrid", "SmallMuseumGrid"]):
    df_env = data[data.env_name == env_name].sort_values(by=['mu'])
    # print(df_env)
    for x, dist_measure in enumerate(dist_measures):
        df = df_env[df_env.dist_measure_name.isin(['null_dist', dist_measure])]
        ax = axes[y, x]
        ax.axhline(y=0, color='black', linestyle=':')
        if env_name in ["MuseumRush", "SmallMuseumGrid"]:
            vases_smashed = - df.vases_smashed
            doors_left_open = [None] * len(mus)
            ax.bar(mus, vases_smashed, facecolor="red", label="Vases smashed", alpha=0.5, width=0.2)
            # Dummy for legend
            ax.bar([0], [0], facecolor="orange", label="Doors left open", alpha=0.5, width=0.2)
        else:
            vases_smashed = [None] * len(mus)
            # print(doors_left_open)
            doors_left_open = - df.doors_left_open
            # print(doors_left_open)
            ax.bar(mus, doors_left_open, facecolor="orange", label="Doors left open", alpha=0.5, width=0.2)
            # ax.bar(mus, doors_left_open) #, c="orange", label="Doors left open")
        if env_name == "SmallMuseumGrid":
            ax.plot([], [], "g:", label="Max. honest task score")
            ax.axhline(y=4.0, color='green', linestyle=':')
        spec_score = df.spec_score
        print(len(df))
        ax.bar(mus, spec_score, facecolor="green", label="Task score", alpha=0.5, width=0.2)
        # ax.plot(mus, spec_score, c="green", label="Task score")
        if y == 2:
            ticks = ["" if i % 2 != 0 else str(mus[i]) for i in range(len(mus))]
            ax.set_xticks(mus, ticks)
        if y == 0:
            dist_measure_dict = {
                "vase_door": "$D_{perf}$",
                "simple": "$D_{simple}$",
                "rgb": "$D_{RGB}$",
            }
            ax.set_title(dist_measure_dict[dist_measure])
        if x == 0:
            ax.set_ylabel(env_name)

handles, labels = axes[-1, -1].get_legend_handles_labels()
fig.legend(handles, labels, loc="center right")
plt.show()
