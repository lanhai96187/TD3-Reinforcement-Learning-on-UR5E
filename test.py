import numpy as np
from UR5E_rl import UR5E_Env
import time
import torch
import torch.nn.functional as F

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x)) * self.action_bound

class TD3_test():
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.device = device

    def take_action(self, state):
        self.actor.eval()
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        action = self.actor(state)
        return action.squeeze().detach().cpu().numpy()


env = UR5E_Env()
action_bound = env.action_space.high[0]
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
hidden_dim = 256
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
agent_test = TD3_test(state_dim, hidden_dim, action_dim, action_bound, device)
agent_test.actor.load_state_dict(torch.load("TD3_UR5E_Actor.pth", map_location=torch.device("cpu")))

env.viewer_init()
while env.viewer.is_running():
    for episode in range(10):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent_test.take_action(state)
            state, reward, done = env.step(action)
            total_reward += reward
            env.render()
            time.sleep(0.01)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

