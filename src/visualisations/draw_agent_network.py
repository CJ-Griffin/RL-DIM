import torch
import torchviz
from src.agents import DQN
from src.rl_utils import load_agent_from_neptune


def draw_agent_network(agent: DQN, eg_input: torch.Tensor):
    model = agent.get_model()
    qs_as = model(eg_input)
    pd = dict(list(model.named_parameters()))
    print(pd)
    dot = torchviz.make_dot(qs_as, params=pd)
    dot.render("torchviz", format="png")


if __name__ == "__main__":
    agent = load_agent_from_neptune(run_name="RLYP-769", ep_no=0)
    draw_agent_network(agent, torch.ones((1,3,9,9)))
