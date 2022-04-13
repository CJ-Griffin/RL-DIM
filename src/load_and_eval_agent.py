from src.custom_envs import SmallMuseumGrid
from src.experiment_manager import run_eval
from src.rl_utils import load_agent_from_neptune, get_env


def main(run_name=None):
    if run_name is None:
        input("Run name?")
    agent = load_agent_from_neptune(run_name, -1)
    env = SmallMuseumGrid()
    env.init_dist_measure("null", 0, 1)
    eval_score, total_info = run_eval(agent, env, 1)
    print(total_info)
    return 0


if __name__ == "__main__":
    main("RLYP-1054")
