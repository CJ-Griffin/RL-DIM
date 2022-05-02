import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

clrs = {"Black": "#001219",
        "Blue": "#005f73",
        # "Green": "#0a9396",
        "Green": "#47682C",
        # "Blue Green": "#94d2bd",
        "Champagne": "#e9d8a6",
        "Yellow Orange": "#ee9b00",
        "Orange": "#ca6702",
        "Orange Red": "#bb3e03",
        "Red": "#ae2012",
        "Dark Red": "#9b2226"}


def plot1():
    mus = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    str_mus = ["0", "$2^0$", "$2^1$", "$2^2$", "$2^3$", "$2^4$", "$2^5$", "$2^6$"]
    # str_mus = [" " + str(int(mu)) + " " for mu in mus]
    columns = ["env_name", "mu", "dist_measure_name", "vases_smashed", "spec_score", "doors_left_open", "sushi_eaten"]
    dist_measures = ["perf", "simple", "rgb"]  # , "rev"]
    env_names = ["MuseumRush", "EasyDoorGrid", "EmptyDirtyRoom", "SmallMuseumGrid"]  # , "SushiGrid"]
    data = pd.read_csv("RL4YP_delta.csv")
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
    fig, axes = plt.subplots(len(env_names),
                             len(dist_measures) + 1,
                             figsize=(9.6, 7.4))
    print(data)
    alpha = 0.8
    for y, env_name in enumerate(env_names):
        df_env = data[data.env_name == env_name].sort_values(by=['mu'])
        ax_im = axes[y, 0]
        im = get_im(env_name)
        ax_im.imshow(im)
        ax_im.set_xticks([])
        ax_im.set_yticks([])
        ax_im.set_ylabel(env_name)
        ax_im.set_frame_on(False)
        for x, dist_measure in enumerate(dist_measures, start=1):
            df = df_env[df_env.dist_measure_name == dist_measure]
            ax = axes[y, x]
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.axhline(y=0, color='black', linewidth=0.8)  # , linestyle=':')
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
                    "rev": "$Rev.$",
                    # "RR": "Relative Reachability"
                }
                ax.set_title(dist_measure_dict[dist_measure])

            # Plot env-specific stuff
            if env_name == "MuseumRush":
                ax.set_ylim(-1.2, 1.2)
                ax.axhline(y=4.6 / 5, color=clrs["Blue"], linestyle=':', label="Max. safe score")
                ax.bar(str_mus, - df.vases_smashed, facecolor=clrs["Red"], label="Vases smashed",
                       alpha=alpha)  # , width=0.2)
                spec_score = df.spec_score / 5
                ax.bar(str_mus, spec_score, facecolor="#084C61", label="Goal reached", alpha=alpha)  # , width=0.2)

            elif env_name == "EmptyDirtyRoom":
                ax.set_ylim(-1.2, 3.6)
                ax.set_yticks([0, 3])
                ax.axhline(y=3.0, color=clrs["Blue"], linestyle=':', label="Max. safe score")
                # There is no side effect on EmptyDirtyRoom!
                spec_score = df.spec_score
                ax.bar(str_mus, spec_score, facecolor=clrs["Blue"], label="Dirt cleaned", alpha=alpha)  # , width=0.2)
                pass

            elif env_name == "EasyDoorGrid":
                ax.set_ylim(-1.2, 1.2)
                ax.set_yticks([-1, 0, 1])
                ax.bar(str_mus, - df.doors_left_open, facecolor=clrs["Yellow Orange"], label="Doors left open",
                       alpha=alpha)  # , width=0.2)
                ax.axhline(y=1.0, color=clrs["Blue"], linestyle=':', label="Max. safe score")
                spec_score = df.spec_score / 5
                ax.bar(str_mus, spec_score, facecolor=clrs["Green"], label="Goal reached", alpha=alpha)  # , width=0.2)

            elif env_name == "SmallMuseumGrid":
                ax.set_ylim(-4.8, 8.8)
                ax.set_yticks([-4, 0, 4, 8])
                ax.bar(str_mus, - df.vases_smashed, facecolor=clrs["Red"], label="Vases smashed",
                       alpha=alpha)  # , width=0.2)
                ax.axhline(y=3.0, color=clrs["Blue"], linestyle=':', label="Max. safe score")
                spec_score = df.spec_score
                ax.bar(str_mus, spec_score, facecolor=clrs["Blue"], label="Dirt cleaned", alpha=alpha)  # , width=0.2)

            elif env_name == "SushiGrid":
                ax.set_ylim(-1.2, 1.2)
                ax.set_yticks([-1, 0, 1])
                ax.bar(str_mus, - 1 + df.sushi_eaten, facecolor=clrs["Orange Red"], label="Sushi interfer.",
                       alpha=alpha)  # , width=0.2)
                spec_score = df.spec_score / 5
                ax.bar(str_mus, spec_score, facecolor=clrs["Green"], label="Goal reached", alpha=alpha)  # , width=0.2)

            else:
                assert NotImplementedError(env_name)

            if x != 1:
                ax.set_yticks([])
            if y != len(env_names) - 1:
                ax.set_xticks([])
            else:
                ax.set_xlabel("$\mu$")

    legend_dict = {}
    for hs, ls in [ax.get_legend_handles_labels() for ax1 in axes for ax in ax1]:
        for h, l in zip(hs, ls):
            legend_dict[l] = h
    labels = list(legend_dict)
    handles = [legend_dict[l] for l in labels]
    fig.legend(handles, labels, loc="center right", fontsize=8)
    plt.tight_layout()
    plt.savefig("results1.pdf")


def get_im(env_name):
    # return plt.imread(f"env_images/{env_name}.png")
    import numpy as np
    from PIL import Image

    path = f"env_images/{env_name}.png"

    img = Image.open(path)

    w = 500
    h = 500
    img.thumbnail((w, h), Image.ANTIALIAS)
    w2, h2 = img.size

    result = Image.new(img.mode, (w, h), "white")
    result.paste(img, (int((w - w2) / 2), int((h - h2) / 2)))

    a = np.asarray(result)
    print("hi!")
    return a


# ================== # ================== # ================== # ================== #

def plot2():
    fig, axes = plt.subplots(1, 3,
                             figsize=(4 * 2 * 1.6, 4))
    # im = plt.imread(f"env_images/RandomMuseumRoom.png")
    # ax_im = axes[0]
    # ax_im.imshow(im)
    # ax_im.tick_params(bottom=False, axis="x")
    # ax_im.tick_params(bottom=False, axis="y")
    # # ax_im.set_xticks([])
    # # ax_im.set_yticks([])
    fig.suptitle("DQN performance on RandomMuseumRoom with $d_{perf}$")
    res = 20

    def get_highs(df):
        return df.nlargest(int(res / 5.0)).mean()

    def get_lows(df):
        return df.nsmallest(int(res / 5.0)).mean()

    nums = ["2599", "2588", "2607"]
    xs = np.arange(start=0, stop=1000000, step=1000)
    mus = [0, 8, 64]

    # reds = [(0.8, 0.0, 0.0), (0.8, 0.4, 0.0), (0.8, 0.6, 0.0)]
    # blues = [(0.0, 0.8, 0.0), (0.0, 0.8, 0.4, 0.0), (0.0, 0.8, 0.6)]

    def plot_spread(ax, xs, ys, col, label, should_max=False):
        if not should_max:
            highs = ys.rolling(res).apply(get_highs)
            mids = ys.rolling(res).mean()
            lows = ys.rolling(res).apply(get_lows)
        else:
            highs = ys.rolling(res).max()
            mids = ys.rolling(res).mean()
            lows = ys.rolling(res).min()
        ax.plot(xs, mids, c=col, label=label)
        ax.fill_between(xs, lows, highs, facecolor=col, alpha=0.5)

    # red = reds[i]
    # blue = blues[i]
    # red = (0.7, 0.1, 0.1)
    # blue = (0.1, 0.7, 0.1)
    blue = clrs["Blue"]
    red = clrs["Red"]
    for i in range(3):
        ax = axes[i]
        ax.plot([0, 1e6], [0, 0], ":", c="black")
        ax.plot([0, 1e6], [3.0, 3.0], ":", c=blue, label="Max. safe score")
        ax.set_ylim(-2.1, 5.1)
        mu, num = mus[i], nums[i]

        rand_data = pd.read_csv(f"DQN_vases_smashed/RLYP-{num}__eval_vases_smashed.csv")
        ys = - rand_data.vases_smashed
        plot_spread(ax, xs, ys, red, label="Vases smashed")

        rand_data = pd.read_csv(f"DQN_spec_score/RLYP-{num}__eval_spec_score.csv")
        ys = rand_data.spec_score
        plot_spread(ax, xs, ys, blue, label="Dirt cleaned")

        ax.set_xlabel("Num. episodes")

    axes[0].set_yticks([-2, 0, 3, 5])
    axes[1].set_yticks([])
    axes[2].set_yticks([])
    axes[0].set_title("$\mu=0$")
    axes[1].set_title("$\mu=8$")
    axes[2].set_title("$\mu=64$")
    axes[2].legend()
    plt.savefig("results2.pdf")


if __name__ == "__main__":
    # plot1()
    plot2()
