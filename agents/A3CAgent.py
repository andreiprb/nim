import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical

from nim.NimLogic import NimLogic
from nim.NimGameState import NimGameState


class A3CNet(nn.Module):
    def __init__(self, num_piles=4, max_pile_size=7):
        super(A3CNet, self).__init__()
        self.num_piles = num_piles
        self.max_pile_size = max_pile_size

        input_size = num_piles * (max_pile_size + 1)

        self.shared = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.policy = nn.Linear(64, num_piles * max_pile_size)

        self.value = nn.Linear(64, 1)

    def forward(self, x):
        x = self.shared(x)
        policy = F.softmax(self.policy(x), dim=1)
        value = self.value(x)
        return policy, value

    def encode_state(self, state):
        encoded = torch.zeros(1, self.num_piles * (self.max_pile_size + 1))
        for i, pile in enumerate(state):
            if pile <= self.max_pile_size:
                encoded[0, i * (self.max_pile_size + 1) + pile] = 1
        return encoded


class Worker(mp.Process):
    def __init__(self, global_model, optimizer, worker_id, num_episodes=100):
        super(Worker, self).__init__()
        self.worker_id = worker_id
        self.num_episodes = num_episodes

        self.local_model = A3CNet()
        self.local_model.load_state_dict(global_model.state_dict())

        self.optimizer = optimizer
        self.global_model = global_model

    def get_valid_action(self, policy, state):
        valid_actions = NimLogic.available_actions(state)

        valid_mask = torch.zeros_like(policy)
        for pile_idx, count in valid_actions:
            action_idx = pile_idx * self.local_model.max_pile_size + (count - 1)
            if action_idx < valid_mask.size(1):
                valid_mask[0, action_idx] = 1

        masked_policy = policy * valid_mask
        if masked_policy.sum() > 0:
            masked_policy = masked_policy / masked_policy.sum()
        else:
            for pile_idx, count in valid_actions:
                action_idx = pile_idx * self.local_model.max_pile_size + (count - 1)
                if action_idx < masked_policy.size(1):
                    masked_policy[0, action_idx] = 1.0 / len(valid_actions)

        m = Categorical(masked_policy)
        action_idx = m.sample()
        log_prob = m.log_prob(action_idx)

        pile_idx = action_idx.item() // self.local_model.max_pile_size
        count = (action_idx.item() % self.local_model.max_pile_size) + 1

        if count > state[pile_idx]:
            action = random.choice(list(valid_actions))
            pile_idx, count = action
            action_idx = torch.tensor(pile_idx * self.local_model.max_pile_size + (count - 1))
            log_prob = torch.log(masked_policy[0, action_idx])

        return (pile_idx, count), log_prob

    def run(self):
        for episode in range(self.num_episodes):
            state = NimGameState()
            done = False
            total_reward = 0

            values = []
            log_probs = []
            rewards = []

            while not done:
                current_state = state.piles.copy()
                state_tensor = self.local_model.encode_state(current_state)

                policy, value = self.local_model(state_tensor)

                action, log_prob = self.get_valid_action(policy, current_state)

                new_state = state.apply_move(action)

                reward = 0
                if new_state.winner is not None:
                    reward = 1 if new_state.winner == state.player else -1
                    done = True

                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)

                state = new_state
                total_reward += reward

            R = 0
            returns = []
            advantages = []

            for r, v in zip(reversed(rewards), reversed(values)):
                R = r + 0.99 * R
                advantage = R - v.item()
                returns.append(R)
                advantages.append(advantage)

            returns.reverse()
            advantages.reverse()

            returns = torch.tensor(returns)
            advantages = torch.tensor(advantages)

            policy_loss = 0
            value_loss = 0

            for log_prob, R, advantage in zip(log_probs, returns, advantages):
                policy_loss -= log_prob * advantage
                value_loss += F.smooth_l1_loss(value, torch.tensor([[R]]))

            total_loss = policy_loss + 0.5 * value_loss

            self.optimizer.zero_grad()
            total_loss.backward()

            for global_param, local_param in zip(self.global_model.parameters(),
                                                 self.local_model.parameters()):
                if global_param.grad is None:
                    global_param.grad = local_param.grad
                else:
                    global_param.grad += local_param.grad

            self.optimizer.step()

            self.local_model.load_state_dict(self.global_model.state_dict())

            if (episode + 1) % 10 == 0:
                print(f"Worker {self.worker_id}, Episode {episode + 1}, Total Reward: {total_reward}")


class A3CAgent:
    def __init__(self, num_piles=4, max_pile_size=7, num_workers=10):
        self.model = A3CNet(num_piles, max_pile_size)
        self.model.share_memory()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.num_workers = num_workers

    def choose_action(self, state, epsilon=None):
        self.model.eval()
        with torch.no_grad():
            state_tensor = self.model.encode_state(state)
            policy, _ = self.model(state_tensor)

            valid_actions = NimLogic.available_actions(state)

            valid_mask = torch.zeros_like(policy)
            for pile_idx, count in valid_actions:
                action_idx = pile_idx * self.model.max_pile_size + (count - 1)
                if action_idx < valid_mask.size(1):
                    valid_mask[0, action_idx] = 1

            masked_policy = policy * valid_mask
            if masked_policy.sum() > 0:
                masked_policy = masked_policy / masked_policy.sum()
            else:
                for pile_idx, count in valid_actions:
                    action_idx = pile_idx * self.model.max_pile_size + (count - 1)
                    if action_idx < masked_policy.size(1):
                        masked_policy[0, action_idx] = 1.0 / len(valid_actions)

            action_idx = torch.argmax(masked_policy).item()
            pile_idx = action_idx // self.model.max_pile_size
            count = (action_idx % self.model.max_pile_size) + 1

            if count > state[pile_idx]:
                return random.choice(list(valid_actions))

            return (pile_idx, count)

    def train(self, num_episodes=1000):
        print(f"Training A3C agent with {self.num_workers} workers for {num_episodes} episodes per worker...")

        workers = []
        for i in range(self.num_workers):
            worker = Worker(self.model, self.optimizer, i, num_episodes // self.num_workers)
            workers.append(worker)
            worker.start()

        for worker in workers:
            worker.join()

        print("Training completed!")