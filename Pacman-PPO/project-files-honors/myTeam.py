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
import heapq

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

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)
    # print("legal actions allowed: ", actions)
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

  def __init__(self, index, timeForComputing = .1 ):
    super().__init__(index, timeForComputing)
    self.numCapsulesRemaining = -1
    self.gotFirstDefenderAfterScare = False
    self.capsulesFinished = False

  def getAStarFirstMove(self, gameState, goalNodeCoords, goalNodeType, recently_ate_capsule=False):
    # NOTE: care about BOTH enemies, NOT just the closest one
    width, height = gameState.data.layout.width, gameState.data.layout.height
    opps = self.getOpponents(gameState)
    myPosition = tuple(int(coord) for coord in gameState.getAgentState(self.index).getPosition())
    goalNodeCoords = tuple(int(coord) for coord in goalNodeCoords)
    opps1_raw_coords = gameState.getAgentPosition(opps[0])
    opps2_raw_coords = gameState.getAgentPosition(opps[1])
    opps1_pos = tuple(int(coord) for coord in opps1_raw_coords) if opps1_raw_coords is not None else None
    opps2_pos = tuple(int(coord) for coord in opps2_raw_coords) if opps2_raw_coords is not None else None
    
    minHeap = []
    uniquenessCounter = 1
    heapq.heappush(minHeap, (0, uniquenessCounter, (0, gameState, [])))
    while len(minHeap) > 0: 
      pop_res = heapq.heappop(minHeap)
      f_n_curr, _, (g_n_curr, currState, listOfMoves) = pop_res
      currStateCoords = currState.getAgentState(self.index).getPosition()
      
      if int(currStateCoords[0]) == int(goalNodeCoords[0]) and int(currStateCoords[1]) == int(goalNodeCoords[1]) or uniquenessCounter > 400:
        if uniquenessCounter == 1:
          return "Stop", f_n_curr
        return listOfMoves[0][0], f_n_curr

      legalActions = currState.getLegalActions(self.index)
      for legalAction in legalActions:
        currSuccessorState = self.getSuccessor(currState, legalAction)
        successorCoords = tuple(int(coord) for coord in currSuccessorState.getAgentState(self.index).getPosition())
        
        h_n = self.getMazeDistance(successorCoords, goalNodeCoords)
        g_n = g_n_curr + 1
        uniquenessCounter += 1
        agentState = gameState.data.agentStates[self.index]
        #print(agentState.scaredTimer)
        if opps1_pos is not None or opps2_pos is not None:
          c1 = opps1_pos is not None and gameState.data.agentStates[opps[0]].scaredTimer > 0
          c2 = opps2_pos is not None and gameState.data.agentStates[opps[1]].scaredTimer > 0
          if c1 or c2 :
            pass 
          elif opps2_pos is not None and gameState.data.agentStates[opps[1]].scaredTimer > 0:
            pass 
          elif goalNodeType == "food" and ((self.red and successorCoords[0] < width // 2 - 2) or (not self.red and successorCoords[0] > width // 2 + 2)):
            g_n = 9999 + uniquenessCounter
          #elif len(self.getCapsules(gameState)) >= self.numCapsulesRemaining:
          else:
            successorCoords = tuple(int(coord) for coord in successorCoords)
            dist1 = self.getMazeDistance(successorCoords, opps1_pos) if opps1_pos is not None else 9999
            dist2 = self.getMazeDistance(successorCoords, opps2_pos) if opps2_pos is not None else 9999
            minDistToEnemy = min(dist1, dist2)
            if opps1_pos is not None and int(opps1_pos[0]) == int(successorCoords[0]) and int(opps1_pos[1]) == int(successorCoords[1]):
              if self.red and successorCoords[0] > width // 2 + 1:
                g_n = 9999 + uniquenessCounter
            if opps2_pos is not None and int(opps2_pos[0]) == int(successorCoords[0]) and int(opps2_pos[1]) == int(successorCoords[1]):
              if self.red and successorCoords[0] > width // 2 + 1:
                g_n = 9999 + uniquenessCounter
            elif minDistToEnemy < 3:
              g_n = 9999 + uniquenessCounter
        heapq.heappush(minHeap, (g_n + h_n , uniquenessCounter, (g_n, currSuccessorState, listOfMoves.copy() + [(legalAction, currSuccessorState)])))
    return None


  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """

    if self.numCapsulesRemaining == -1:
      self.numCapsulesRemaining = len(self.getCapsules(gameState))


    width, height = gameState.data.layout.width, gameState.data.layout.height
    myPosition = tuple(int(coord) for coord in gameState.getAgentState(self.index).getPosition())
    foodList = self.getFood(gameState).asList()
    wallSet = set(gameState.data.layout.walls.asList())
    borderList = [(width // 2 - 1, y) for y in range(height)] if self.red else [(width // 2 + 1, y) for y in range(height)] 
    numCarrying = gameState.getAgentState(self.index).numCarrying
    bestAction = None
    numCapsulesRemaining = len(self.getCapsules(gameState))

    if numCapsulesRemaining < self.numCapsulesRemaining:
      if numCarrying < 5:
        myPosition = tuple(int(coord)for coord in myPosition)
        foodListGoalsSorted = sorted(foodList, key=lambda goalNode: self.evaluateGoalNodeScore(gameState, goalNode, self.getMazeDistance(myPosition, tuple(int(coord) for coord in goalNode)), "food"), reverse=True)
        for goalNode in foodListGoalsSorted:
          bestAction, totalCostOfOptimalPath = self.getAStarFirstMove(gameState, goalNode, "food", True)
          if totalCostOfOptimalPath < 9999:
            return bestAction 
      else:
        self.numCapsulesRemaining = numCapsulesRemaining


    numCapsulesRemaining = len(self.getCapsules(gameState))
    if numCapsulesRemaining > 0: # DO NOT CHANGE TO ELIF
      goalNode = self.getCapsules(gameState)[0]
      bestAction, _ = self.getAStarFirstMove(gameState, goalNode, "capsule")
      return bestAction 
    
    if gameState.getAgentState(self.index).numCarrying < 2:
      foodListGoalsSorted = sorted(foodList, key=lambda goalNode: self.evaluateGoalNodeScore(gameState, goalNode, self.getMazeDistance(myPosition, tuple(int(coord) for coord in goalNode)), "food"), reverse=True)
      for goalNode in foodListGoalsSorted:
        bestAction, totalCostOfOptimalPath = self.getAStarFirstMove(gameState, goalNode, "food")
        if totalCostOfOptimalPath < 9999:
          return bestAction 
    else: 
      myPositionCoordsInt = (int(myPosition[0]), int(myPosition[1]))
      borderListWithoutWalls = [borderPosition for borderPosition in borderList if borderPosition not in wallSet]
      borderListSorted = sorted(borderListWithoutWalls, key=lambda goalNode: self.evaluateGoalNodeScore(gameState, goalNode, self.getMazeDistance(myPositionCoordsInt, (int(goalNode[0]), int(goalNode[1]))), "food"), reverse=True)
      for goalNode in borderListSorted:
        bestAction, totalCostOfOptimalPath = self.getAStarFirstMove(gameState, goalNode, "border")
        if totalCostOfOptimalPath < 9999:
          return bestAction 
        
    return bestAction 
  
  def evaluateGoalNodeScore(self, gameState, goalNodeCoords, numMovesToGoalNode, nodeType):
    features = self.getGoalFeatures(gameState, goalNodeCoords, numMovesToGoalNode, nodeType)
    weights = self.getGoalWeights()
    return features * weights

  def getGoalFeatures(self, gameState, goalNodeCoords, numMovesToGoalNode, nodeType):
    # for both food and border, factor in enemy proximity and number of moves to get to goal node
    features = util.Counter()
    features["distanceToGoalNode"] = numMovesToGoalNode
    
    return features

  def getGoalWeights(self):
    return {"distanceToGoalNode": -2}


  

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def __init__(self, index, timeForComputing=.1):
    super(DefensiveReflexAgent, self).__init__(index, timeForComputing)
    self.coordSelectionIndex = 0
    self.coordSelectionDirection = 1
    self.waitingAreaCandidateCoords = None


  def getAStarFirstMove(self, gameState, goalNodeCoords):
    opps = self.getOpponents(gameState)
    width, height = gameState.data.layout.width, gameState.data.layout.height
    goalNodeCoords = tuple(int(coord) for coord in goalNodeCoords)
    opps1_raw_coords = gameState.getAgentPosition(opps[0])
    opps2_raw_coords = gameState.getAgentPosition(opps[1])
    opps1_pos = tuple(int(coord) for coord in opps1_raw_coords) if opps1_raw_coords is not None else None
    opps2_pos = tuple(int(coord) for coord in opps2_raw_coords) if opps2_raw_coords is not None else None
    dist1 = self.getMazeDistance(opps1_pos, goalNodeCoords) if opps1_pos is not None else 9999
    dist2 = self.getMazeDistance(opps2_pos, goalNodeCoords) if opps2_pos is not None else 9999
    distanceToClosestEnemy = min(dist1, dist2)
    closestEnemyCoords = opps1_pos if distanceToClosestEnemy == dist1 else opps2_pos

    minHeap = []
    uniquenessCounter = 1
    heapq.heappush(minHeap, (0, uniquenessCounter, (0, gameState, [])))
    while len(minHeap) > 0: 
      pop_res = heapq.heappop(minHeap)
      _, _, (g_n_curr, currState, listOfMoves) = pop_res
      
      currStateCoords = currState.getAgentState(self.index).getPosition()
      if int(currStateCoords[0]) == int(goalNodeCoords[0]) and int(currStateCoords[1]) == int(goalNodeCoords[1]) or uniquenessCounter > 400: #or uniquenessCounter > 400:
        if uniquenessCounter == 1:
          return "Stop"
        return listOfMoves[0][0] 

      legalActions = currState.getLegalActions(self.index)
      for legalAction in legalActions:
        currSuccessorState = self.getSuccessor(currState, legalAction)
        successorCoords = tuple(int(coord) for coord in currSuccessorState.getAgentState(self.index).getPosition())
        
        h_n = self.getMazeDistance(successorCoords, goalNodeCoords)
        g_n = g_n_curr + 1
        uniquenessCounter += 1

        if (self.red and int(successorCoords[0]) > width // 2 + 1) or (not self.red and int(successorCoords[0] < width // 2 - 1)):
          g_n = 9999 + uniquenessCounter


        heapq.heappush(minHeap, (h_n + g_n, uniquenessCounter, (g_n, currSuccessorState, listOfMoves.copy() + [(legalAction, currSuccessorState)])))
    return None
  

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)
    width, height = gameState.data.layout.width, gameState.data.layout.height
    if self.waitingAreaCandidateCoords is None:
      self.waitingAreaCandidateCoords = []
      wallsSet = set(gameState.data.layout.walls.asList())
      if self.red:
        for j in range(height // 6, 5 * height // 6):
          for k in range(1, 6):
            i = width // 2 - k
            if (i,j) not in wallsSet:
              self.waitingAreaCandidateCoords.append((i,j))
      else:
        for j in range(height // 6, 5 * height // 6):
          for k in range(1, 6):
            i = width // 2 + k
            if (i,j) not in wallsSet:
              self.waitingAreaCandidateCoords.append((i,j))
    opps = self.getOpponents(gameState)
    opps1_raw_coords = gameState.getAgentPosition(opps[0])
    opps2_raw_coords = gameState.getAgentPosition(opps[1])
    opps1_pos = tuple(int(coord) for coord in opps1_raw_coords) if opps1_raw_coords is not None else None
    opps2_pos = tuple(int(coord) for coord in opps2_raw_coords) if opps2_raw_coords is not None else None
    myPosition = tuple(int(coord) for coord in gameState.getAgentState(self.index).getPosition())
    dist1 = self.getMazeDistance(opps1_pos, myPosition) if opps1_pos is not None else 9999
    dist2 = self.getMazeDistance(opps2_pos, myPosition) if opps2_pos is not None else 9999
    distanceToClosestEnemy = min(dist1, dist2)
    closestEnemyCoords = opps1_pos if distanceToClosestEnemy == dist1 else opps2_pos
    
    if closestEnemyCoords is not None and (int(myPosition[0]) != int(closestEnemyCoords[0]) or int(myPosition[1]) != int(closestEnemyCoords[1])) and ((self.red and closestEnemyCoords[0] < width // 2 - 2) or (not self.red and closestEnemyCoords[0] > width // 2 + 2)):
      e_y = min(closestEnemyCoords[1], width // 2 - 1) if self.red else max(closestEnemyCoords[1], width // 2 + 1)
      bestAction = self.getAStarFirstMove(gameState, (closestEnemyCoords[0], e_y))
      return bestAction 
    else:
      waitingCoordsSelection = self.waitingAreaCandidateCoords[self.coordSelectionIndex]
      if self.coordSelectionIndex == len(self.waitingAreaCandidateCoords) - 1:
        self.coordSelectionDirection = -1
      elif self.coordSelectionIndex == 0:
        self.coordSelectionDirection = 1
      self.coordSelectionIndex += self.coordSelectionDirection
      
      bestAction = self.getAStarFirstMove(gameState, waitingCoordsSelection)
      return bestAction
    






