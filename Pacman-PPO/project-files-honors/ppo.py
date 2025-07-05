import torch
import subprocess
import random
import json 
from torch import nn, optim
import os 

GAE_PARAM = 0.95
DISCOUNT_FACTOR = 0.99
NUM_ROLLOUTS = 10000
NUM_EPOCHS_PER_ROLLOUT = 10
NUM_SAMPLED_TRANSITIONS = 128
NUM_GAMES_PER_BATCH = 4
VALUE_NET_LR = 3e-4
POLICY_NET_LR = 3e-4
EPSILON_CLIP = 0.2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class NeuralNet(nn.Module):
  def __init__(self, in_dim, hidden_dim, out_dim, include_softmax=True):
    super(NeuralNet, self).__init__()
    self.linear1 = nn.Linear(in_dim, hidden_dim)
    self.relu = nn.ReLU()
    self.linear2 = nn.Linear(hidden_dim, out_dim)
    if include_softmax: self.softmax = nn.Softmax(dim=-1)
    else:
        self.softmax = None
  
  def forward(self, normalized_inputs):
    normalized_inputs = self.relu(self.linear1(normalized_inputs))
    outputs = self.linear2(normalized_inputs)
    if self.softmax is None: 
        return outputs
    output_probs = self.softmax(outputs)
    return output_probs


class PPO:
    def __init__(self):
        self.mse = nn.MSELoss()
        self.policy_net = NeuralNet(in_dim=39, hidden_dim=64, out_dim=5)
        self.value_net = NeuralNet(in_dim=39, hidden_dim=64, out_dim=1, include_softmax=False)
        self.policy_net.to(DEVICE)
        self.value_net.to(DEVICE)
        self.num_sampled_transitions = NUM_SAMPLED_TRANSITIONS
        self.num_games_per_batch = NUM_GAMES_PER_BATCH
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=POLICY_NET_LR)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=VALUE_NET_LR)
        self.action_to_idx = {
            "North": 0,
            "South": 1, 
            "East": 2, 
            "West": 3,
            "Stop": 4
        }


    def runBatchOfGames(self):
        command = ["python", "capture.py", "-r", "myTeamDraft2.py", "-b", "baselineTeam3.py"]
        transition_lists = []
        
        for _ in range(NUM_GAMES_PER_BATCH):
            # run game
            _ = subprocess.run(
                command,
                capture_output=True, # capture the output of running the command
                text=True, # return output as string, NOT binary
                check=True # raise error if fail
            )
            with open("action_reward_list.json", "r") as file:
               game_transition_list = json.load(file)

            # sample 32 transitions from each game (128 total)
            sampled_indices = random.sample(range(len(game_transition_list)), k=int(NUM_SAMPLED_TRANSITIONS / NUM_GAMES_PER_BATCH))
            transition_lists += [(idx, game_transition_list[idx:]) for idx in range(len(game_transition_list)) if idx in sampled_indices]
            os.remove("action_reward_list.json")

        policy_net_state_dict = torch.load("policy_net_weights.pth")
        self.policy_net.load_state_dict(policy_net_state_dict)

        return transition_lists

    def calculateTransitionAdvantage(self, transition_list):
        advantage = 0
        running_reward = 0
        for k in range(len(transition_list)): 
            _, _, reward, policy_inputs, _ = transition_list[k]  
            running_reward += (DISCOUNT_FACTOR ** k) * reward
            curr_val_estimate = self.value_net(torch.tensor(policy_inputs, dtype=torch.float32, device=DEVICE))
            unweighted_advantage = running_reward
            if k < len(transition_list) - 1: 
                future_val_estimate = self.value_net(torch.tensor(transition_list[k+1][-2], dtype=torch.float32, device=DEVICE)) 
                unweighted_advantage += DISCOUNT_FACTOR * future_val_estimate - curr_val_estimate
            advantage += ((GAE_PARAM * DISCOUNT_FACTOR) ** k) * unweighted_advantage
        return advantage, running_reward, transition_list[0][-2], transition_list[0][-1] # policy inputs for actual sampled action

    def trainPPO(self):
        for rollout in range(NUM_ROLLOUTS):
            print("STARTING ROLLOUT ", rollout+1)
            transition_lists = self.runBatchOfGames()
            self.policy_net.train()
            self.value_net.train()
            for epoch in range(NUM_EPOCHS_PER_ROLLOUT):
                print("EPOCH ", epoch)
                random.shuffle(transition_lists)
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()

                advantage_tuples = [self.calculateTransitionAdvantage(transition_subset[1]) for transition_subset in transition_lists]
                advantages = torch.tensor([advantage_tuple[0] for advantage_tuple in advantage_tuples], dtype=torch.float32, device=DEVICE)
                running_rewards = torch.tensor([advantage_tuple[1] for advantage_tuple in advantage_tuples], dtype=torch.float32, device=DEVICE)
                neural_net_inputs = torch.tensor([advantage_tuple[2] for advantage_tuple in advantage_tuples], dtype=torch.float32, device=DEVICE)
                valid_actions = [advantage_tuple[3] for advantage_tuple in advantage_tuples]

                actions = [transition_subset[1][0][0] for transition_subset in transition_lists]
                action_indices = torch.tensor([self.action_to_idx[action] for action in actions], dtype=torch.int64, device=DEVICE)
                old_action_probs = torch.tensor([transition_subset[1][0][1] for transition_subset in transition_lists], dtype=torch.float32, device=DEVICE)

                action_probs = self.policy_net(neural_net_inputs) # (T, 5)
                possible_actions = ["North", "South", "East", "West", "Stop"]
                mask = []
                for valid_action_list in valid_actions:
                    list_mask = []
                    for possible_action in possible_actions:
                        list_mask.append(0 if possible_action in valid_action_list else -1e9)
                    mask.append(list_mask)
                mask = torch.tensor(mask, dtype=torch.float32, device=DEVICE)
                action_probs = nn.functional.softmax(action_probs + mask, dim=1)
                selected_action_probs = action_probs.gather(dim=1, index=action_indices.unsqueeze(1)).squeeze(1)
                action_prob_ratios = selected_action_probs / old_action_probs
                unclipped_ratio_advantages = action_prob_ratios * advantages
                clipped_ratio_advantages = torch.clip(unclipped_ratio_advantages, min=1-EPSILON_CLIP, max=1+EPSILON_CLIP) * advantages
                final_ratio_advantages = torch.minimum(unclipped_ratio_advantages, clipped_ratio_advantages)
                L_clip = -1. / len(transition_lists) * torch.sum(final_ratio_advantages)
                L_clip.backward()
                self.policy_optimizer.step()

                reward_estimates = self.value_net(neural_net_inputs).squeeze(1)
                value_loss = self.mse(reward_estimates, running_rewards)
                value_loss.backward()
                self.value_optimizer.step()

            # write the new policy network weights to weights file 
            torch.save(self.policy_net.state_dict(), "policy_net_weights.pth")


ppo_trainer = PPO()
ppo_trainer.trainPPO()



            

            
        
            
        


        



