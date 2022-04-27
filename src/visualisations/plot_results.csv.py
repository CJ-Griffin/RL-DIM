import pandas as pd
import matplotlib.pyplot as plt

mus = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
str_mus = [" " + str(int(mu)) + " " for mu in mus]
# columns = ["env_name", "mu", "eVases (last)","eSushi (last)","eDoors (last)","eSpec (last)"]
columns = ["env_name", "mu", "dist_measure_name", "vases_smashed", "spec_score", "doors_left_open", "sushi_eaten"]
dist_measures = ["perf", "simple", "rgb", "rev"]
env_names = ["MuseumRush", "EmptyDirtyRoom", "EasyDoorGrid", "SmallMuseumGrid", "SushiGrid"]

data = pd.read_csv("RL4YP_beta.csv")
print(data.columns)
renames = dict([(f"parameters/{name}", name) for name in columns])
data = data.rename(columns=renames)
print(data.columns)
renames = dict([(f"eval/{name} (last)", name) for name in columns])
data = data.rename(columns=renames)
print(data.columns)
data = data.filter(columns)
print(data.columns)
print(data.env_name.unique())
fig, axes = plt.subplots(len(env_names), len(dist_measures), sharex=True, sharey=True)
print(data)
for y, env_name in enumerate(env_names):
    df_env = data[data.env_name == env_name].sort_values(by=['mu'])
    for x, dist_measure in enumerate(dist_measures):
        df = df_env[df_env.dist_measure_name == dist_measure]

        ax = axes[y, x]
        ax.axhline(y=0, color='black', linestyle=':')
        # if y == 2:
        #     ticks = ["" if i % 2 != 0 else str(mus[i]) for i in range(len(mus))]
        #     ax.set_xticks(mus, ticks)
        # ax.set_xscale('log')

        # Add labels to axes
        if y == 0:
            dist_measure_dict = {
                "perf": "$D_{perf}$",
                "simple": "$D_{simple}$",
                "rgb": "$D_{RGB}$",
                "rev": "$D_{rev}$",
            }
            ax.set_title(dist_measure_dict[dist_measure])
        if x == 0:
            ax.set_ylabel(env_name)

        # Plot env-specific stuff
        if env_name == "MuseumRush":
            vases_smashed = - df.vases_smashed
            ax.bar(str_mus, - df.vases_smashed, facecolor="red", label="Vases smashed", alpha=0.5, width=0.2)
            spec_score = df.spec_score / 5
            ax.bar(str_mus, spec_score, facecolor="green", label="% specified return", alpha=0.5, width=0.2)

        elif env_name == "EmptyDirtyRoom":
            # There is no side effect on EmptyDirtyRoom!
            spec_score = df.spec_score / 3
            ax.bar(str_mus, spec_score, facecolor="green", label="% specified return", alpha=0.5, width=0.2)
            pass

        elif env_name == "EasyDoorGrid":
            ax.bar(str_mus, - df.doors_left_open, facecolor="orange", label="Doors left open", alpha=0.5, width=0.2)
            spec_score = df.spec_score / 3
            ax.bar(str_mus, spec_score, facecolor="green", label="% specified return", alpha=0.5, width=0.2)

        elif env_name == "SmallMuseumGrid":
            ax.bar(str_mus, - df.vases_smashed, facecolor="red", label="Vases smashed", alpha=0.5, width=0.2)
            ax.plot([0], [0], "g:", label="Max. honest task score")
            ax.axhline(y=1.0, color='green', linestyle=':')
            spec_score = df.spec_score / 3
            ax.bar(str_mus, spec_score, facecolor="green", label="% specified return", alpha=0.5, width=0.2)

        elif env_name == "SushiGrid":
            ax.bar(str_mus, - df.sushi_eaten, facecolor="magenta", label="Sushi Eaten", alpha=0.5, width=0.2)
            spec_score = df.spec_score / 5
            ax.bar(str_mus, spec_score, facecolor="green", label="% specified return", alpha=0.5, width=0.2)

        else:
            assert NotImplementedError(env_name)

handles, labels = axes[-1, -1].get_legend_handles_labels()
fig.legend(handles, labels, loc="center right")
plt.show()
