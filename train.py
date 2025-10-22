import random
import gym
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import collections
from UR5E_rl import UR5E_Env

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)

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


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1) # 拼接状态和动作
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)

class TD3:
    ''' TD3算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, action_noise,
                 actor_lr, critic_lr, tau, gamma, update_delay, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        # 初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        # 初始化目标策略网络并设置和策略相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.action_bound = action_bound  # 动作限幅
        self.action_noise = action_noise  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = action_dim
        self.update_delay = update_delay    # Actor网络延迟更新，Critic更新多轮后再更新Actor
        self.update_iteration = 0       # 更新轮数
        self.device = device

    def take_action(self, state):
        self.actor.eval()
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        action = self.actor(state)
        # 给动作添加噪声，增加探索
        noise = torch.tensor(np.random.normal(loc=0.0, scale=self.action_noise), dtype=torch.float).to(self.device)
        action = torch.clamp(action + noise, -self.action_bound, self.action_bound)
        return action.squeeze().detach().cpu().numpy()

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            next_q_values_1 = self.target_critic_1(next_states, next_actions)
            next_q_values_2 = self.target_critic_2(next_states, next_actions)
            q_targets = rewards + self.gamma * torch.min(next_q_values_1, next_q_values_2) * (1 - dones)

        critic_1_loss = torch.mean(F.mse_loss(self.critic_1(states, actions), q_targets))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        critic_2_loss = torch.mean(F.mse_loss(self.critic_2(states, actions), q_targets))
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        self.update_iteration += 1  # 延迟更新Actor网络
        if self.update_iteration % self.update_delay == 0:
            actor_loss = -torch.mean(self.critic_1(states, self.actor(states)))
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
            self.soft_update(self.critic_1, self.target_critic_1)  # 软更新价值网络
            self.soft_update(self.critic_2, self.target_critic_2)

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                           'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

env = UR5E_Env()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

actor_lr = 1e-4
critic_lr = 1e-3
num_episodes = 1000
hidden_dim = 256
gamma = 0.99
tau = 0.005  # 软更新参数
buffer_size = 1000000
minimal_size = 1000
batch_size = 128
action_noise = 0.01
update_delay = 1
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

replay_buffer = ReplayBuffer(buffer_size)
agent = TD3(state_dim, hidden_dim, action_dim, action_bound, action_noise, actor_lr, critic_lr,
            tau, gamma, update_delay, device)

agent.actor.load_state_dict(torch.load("TD3_UR5E_Actor.pth", map_location=torch.device("cpu")))
agent.critic_1.load_state_dict(torch.load("TD3_UR5E_Critic1.pth", map_location=torch.device("cpu")))
agent.critic_2.load_state_dict(torch.load("TD3_UR5E_critic_2.pth", map_location=torch.device("cpu")))

return_list = train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('TD3 on UR5E')
plt.show()

torch.save(agent.actor.state_dict(), 'TD3_UR5E_Actor.pth')
