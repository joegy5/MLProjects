# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint

import torch
from torch import nn
import os

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class OffensePolicyNetwork(nn.Module):
  def __init__(self, in_dim, hidden_dim, out_dim):
    super(OffensePolicyNetwork, self).__init__()
    self.linear1 = nn.Linear(in_dim, hidden_dim)
    self.relu = nn.ReLU()
    self.linear2 = nn.Linear(hidden_dim, out_dim)
    self.softmax = nn.Softmax(dim=-1)
  
  def forward(self, normalized_inputs):
    normalized_inputs = self.relu(self.linear1(normalized_inputs))
    normalized_inputs = self.linear2(normalized_inputs)
    output_probs = self.softmax(normalized_inputs)
    return output_probs


class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """

  def __init__(self, index, timeForComputing = .1):
    super(ReflexCaptureAgent, self).__init__(index, timeForComputing)
    self.policy_net = OffensePolicyNetwork(in_dim=39, hidden_dim=64, out_dim=5)

 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def compute_components_dist(self, agent_pos, other_pos):
    if other_pos is None: return [-1.0, -1.0]
    return [other_pos[0] - agent_pos[0], other_pos[1] - agent_pos[1]]

  def getPolicyInputs(self, gameState):  
    inputs = []
    agent_pos = gameState.getAgentPosition(self.index)
    agent_pos = (int(agent_pos[0]), int(agent_pos[1]))
    foodSet = set(self.getFood(gameState).asList())
    height, width = gameState.data.layout.height, gameState.data.layout.width

    # Add all food positions within 5x5 space (binary values --> no need to normalize)
    surrounding_food = []
    #print("reached here")
    for x in range(agent_pos[0] - 2, agent_pos[0] + 3):
      for y in range(agent_pos[1] - 2, agent_pos[1] + 3):
        if(x != agent_pos[0] or y != agent_pos[1]):
          surrounding_food.append(int((x, y) in foodSet))
    inputs += surrounding_food

    # Add agent coordinates (relative to nearest 3 food pellets, power pellet, and BOTH enemies) (normalized by total width of grid)
    food_dists = list(sorted(map(lambda food_coord: self.compute_components_dist(agent_pos, food_coord), self.getFood(gameState).asList()), key=lambda comp_dist: abs(comp_dist[0]) + abs(comp_dist[1])))[:3] 
    closest_power_pellet_dist = list(min(map(lambda p_pellet_coord: self.compute_components_dist(agent_pos, p_pellet_coord), self.getCapsules(gameState)), key=lambda comp_dist: abs(comp_dist[0]) + abs(comp_dist[1])))
    opp_dists = list(map(lambda opp_idx: self.compute_components_dist(agent_pos, gameState.getAgentPosition(opp_idx)), self.getOpponents(gameState)))
    relative_coords = []
    relative_coords += [coord_component for food_coord in food_dists for coord_component in food_coord]
    relative_coords += closest_power_pellet_dist
    relative_coords += [coord_component for opp_coord_dist in opp_dists for coord_component in opp_coord_dist]
    for coord_comp_idx in range(len(relative_coords)):
      if coord_comp_idx % 2 == 0: 
        inputs.append(relative_coords[coord_comp_idx] / width)
      else: 
        inputs.append(relative_coords[coord_comp_idx] / height)


    # Add agent's horizontal distance to border (normalized by total width of grid)
    inputs.append(.5 - agent_pos[0] / width)

    # Add power pellet timer for each enemy (normalized by total timer time)
    for opp_idx in self.getOpponents(gameState):
      if opp_idx is not None:
        opp_scared_timer = gameState.data.agentStates[opp_idx].scaredTimer
      else:
        opp_scared_timer = 0.0
      inputs.append(opp_scared_timer / 40.0)
    
    return torch.tensor(inputs, dtype=torch.float32).unsqueeze(0) # (1, n_dim)


  def calculate_reward(self, gameState, action):
    food_list = self.getFood(gameState).asList()
    food_set = set(food_list)
    dir_dict = {"North": (0, 1),
                "South": (0, -1),
                "East":  (1, 0),
                "West":  (-1, 0),
                "Stop":  (0, 0)}
    old_pos = gameState.getAgentPosition(self.index)
    new_x = old_pos[0] + dir_dict[action][0]
    new_y = old_pos[1] + dir_dict[action][1]
    new_pos = (new_x, new_y)
    width = gameState.data.layout.width
    enemies = self.getOpponents(gameState)
    enemy_coords = []
    capsules = self.getCapsules(gameState)
    for enemy_idx in enemies:
      if gameState.getAgentPosition(enemy_idx) is not None:
        enemy_coords.append((enemy_idx, gameState.getAgentPosition(enemy_idx)))
    reward = 0

    # SPARSE REWARDS ------------------------------------
    # Capture food: +3
    if new_pos in food_set:
      reward += 3
    # Return food: +10 x min(# carrying, 3)
    if new_pos[0] < width // 2:
      reward += 10 * min(3, gameState.data.agentStates[self.index].numCarrying)
    # Eat scared enemy: +2 (-5 if not scared --> eaten by enemy)
    for enemy in enemy_coords:
      if enemy[1] == new_pos:
        if gameState.data.agentStates[enemy[0]].scaredTimer > 0:
          reward += 2
        else:
          reward -= 5
    # capture power pellet: +5
    if new_pos in capsules:
      reward += 5
    
    # DENSE REWARDS -------------------------------------
    # any move: -0.01 (encourage efficiency)
    reward -= 0.01
    # move towards top 3 closest foods: +0.05 for each food closer to
    top_3_closest_foods = sorted(food_list, key=lambda food_coord: self.getMazeDistance(old_pos, food_coord))[:3]
    for food_coord in top_3_closest_foods:
      old_dist = self.getMazeDistance(old_pos, food_coord)
      new_dist = self.getMazeDistance(new_pos, food_coord)
      reward += 0.05 * (new_dist < old_dist)
    
    # move towards (nearby) enemy: -0.1 for each enemy moving closer to
    for enemy in enemy_coords:
      old_dist = self.getMazeDistance(old_pos, enemy[1])
      new_dist = self.getMazeDistance(new_pos, enemy[1])
      reward -= 0.1 * (new_dist < old_dist)
    
    # Move towards power pellet: +0.2
    closest_power_pellet = min(capsules, key=lambda p_pellet: self.getMazeDistance(old_pos, p_pellet))
    old_dist = self.getMazeDistance(old_pos, closest_power_pellet)
    new_dist = self.getMazeDistance(new_pos, closest_power_pellet)
    reward += 0.2 * (new_dist < old_dist)
    # Move towards border (if on enemy side and carrying food): +0.1
    if gameState.data.agentStates[self.index].numCarrying > 0:
      if new_pos[0] < old_pos[0]: # NOTE: only applies if on red time
        reward += 0.1

    return reward



  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    policy_inputs = self.getPolicyInputs(gameState)
    print("POLICY INPUTS SIZE: ", policy_inputs.size())
    print("POLICY_INPUTS: ", policy_inputs)
    width, height = gameState.data.layout.width, gameState.data.layout.height
    
    actions = gameState.getLegalActions(self.index)
    # print(actions)




    if os.path.exists("policy_net_weights.pth"):
      policy_net_state_dict = torch.load("policy_net_weights.pth")
      self.policy_net.load_state_dict(policy_net_state_dict)
    # Store weights to perform policy optimization in separate file
    torch.save(self.policy_net.state_dict(), "policy_net_weights.pth")

    # get probabilities for the 5 possible actions
    policy_inputs = self.getPolicyInputs(gameState) # tensor (1, n_dim)
    action_probs = self.policy_net(policy_inputs)[0, :]
    
    # mask invalid actions
    valid_actions = gameState.getLegalActions(self.index)
    possible_actions = ["North", "South", "East", "West", "Stop"]
    mask = []
    for possible_action in possible_actions:
      mask.append(0 if possible_action in valid_actions else -1e9)
    mask = torch.tensor(mask, dtype=torch.float32)
    print("ACTION PROBS (PRE MASK): ", action_probs)
    print("MASK: ", mask)
    masked_action_probs = nn.functional.softmax(action_probs + mask, dim=0)

    # choose action and calculate reward
    chosen_action_idx = torch.multinomial(masked_action_probs, num_samples=1, replacement=True)
    chosen_action = possible_actions[chosen_action_idx]
    chosen_action_prob = masked_action_probs[chosen_action_idx]
    action_reward = self.calculate_reward(gameState, chosen_action)






    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    
    
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
