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
    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      bestAction = actions[0]  # Default to first action in case no better action is found
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start, pos2)
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
    # Make sure goal coordinates are valid
    width, height = gameState.data.layout.width, gameState.data.layout.height
    wallGrid = gameState.getWalls()
    
    # Convert goal coordinates to integers if they aren't already
    goalNodeCoords = (int(goalNodeCoords[0]), int(goalNodeCoords[1]))
    
    # Check if goal position is within bounds and not a wall
    if (goalNodeCoords[0] < 0 or goalNodeCoords[0] >= width or 
        goalNodeCoords[1] < 0 or goalNodeCoords[1] >= height or
        wallGrid[goalNodeCoords[0]][goalNodeCoords[1]]):
        # Return a safe default action
        return "Stop", float('inf')
    
    # Get enemy information
    opps = self.getOpponents(gameState)
    myPosition = gameState.getAgentState(self.index).getPosition()
    myPosition = (int(myPosition[0]), int(myPosition[1]))
    
    # Safely get opponent positions
    opps1_raw_coords = gameState.getAgentPosition(opps[0])
    opps2_raw_coords = gameState.getAgentPosition(opps[1])
    opps1_pos = (int(opps1_raw_coords[0]), int(opps1_raw_coords[1])) if opps1_raw_coords is not None else None
    opps2_pos = (int(opps2_raw_coords[0]), int(opps2_raw_coords[1])) if opps2_raw_coords is not None else None
    
    minHeap = []
    uniquenessCounter = 1
    heapq.heappush(minHeap, (0, uniquenessCounter, (0, gameState, [])))
    
    while len(minHeap) > 0: 
      pop_res = heapq.heappop(minHeap)
      f_n_curr, _, (g_n_curr, currState, listOfMoves) = pop_res
      currStateCoords = currState.getAgentState(self.index).getPosition()
      currStateCoords = (int(currStateCoords[0]), int(currStateCoords[1]))
      
      # Check if we've reached the goal or search limit
      if currStateCoords == goalNodeCoords or uniquenessCounter > 400:
        if uniquenessCounter == 1:
          return "Stop", f_n_curr
        return listOfMoves[0][0], f_n_curr

      legalActions = currState.getLegalActions(self.index)
      for legalAction in legalActions:
        # Skip "Stop" action to improve efficiency
        if legalAction == "Stop" and len(legalActions) > 1:
            continue
            
        try:
            currSuccessorState = self.getSuccessor(currState, legalAction)
            successorCoords = currSuccessorState.getAgentState(self.index).getPosition()
            successorCoords = (int(successorCoords[0]), int(successorCoords[1]))
            
            h_n = self.getMazeDistance(successorCoords, goalNodeCoords)
            g_n = g_n_curr + 1
            uniquenessCounter += 1
            
            # Avoid enemies unless we've eaten a capsule
            if (opps1_pos is not None or opps2_pos is not None) and not recently_ate_capsule:
              if len(self.getCapsules(gameState)) >= self.numCapsulesRemaining:
                dist1 = self.getMazeDistance(successorCoords, opps1_pos) if opps1_pos is not None else 9999
                dist2 = self.getMazeDistance(successorCoords, opps2_pos) if opps2_pos is not None else 9999
                minDistToEnemy = min(dist1, dist2)
                
                # Avoid enemy positions
                if (opps1_pos is not None and successorCoords == opps1_pos) or (opps2_pos is not None and successorCoords == opps2_pos):
                  if self.red and successorCoords[0] > width // 2 + 1:
                    g_n = 9999 + uniquenessCounter
                # Avoid positions close to enemies
                elif minDistToEnemy < 3:
                  g_n = 9999 + uniquenessCounter
            # Stay in safe territory when looking for food
            elif goalNodeType == "food":
                if (self.red and successorCoords[0] < width // 2 - 2) or (not self.red and successorCoords[0] > width // 2 + 2):
                    g_n = 9999 + uniquenessCounter
                    
            heapq.heappush(minHeap, (g_n + h_n, uniquenessCounter, (g_n, currSuccessorState, listOfMoves.copy() + [(legalAction, currSuccessorState)])))
        except Exception as e:
            # Skip this action if there's an error
            continue
            
    # If no path found, return Stop
    return "Stop", float('inf')


  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    # Initialize the number of capsules if this is the first turn
    if self.numCapsulesRemaining == -1:
      self.numCapsulesRemaining = len(self.getCapsules(gameState))

    # Get game information
    width, height = gameState.data.layout.width, gameState.data.layout.height
    myPosition = gameState.getAgentState(self.index).getPosition()
    myPosition = (int(myPosition[0]), int(myPosition[1]))
    foodList = self.getFood(gameState).asList()
    wallSet = set(gameState.data.layout.walls.asList())
    borderList = [(width // 2 - 1, y) for y in range(height) if not gameState.hasWall(width // 2 - 1, y)] if self.red else [(width // 2 + 1, y) for y in range(height) if not gameState.hasWall(width // 2 + 1, y)]
    numCarrying = gameState.getAgentState(self.index).numCarrying
    bestAction = None
    numCapsulesRemaining = len(self.getCapsules(gameState))

    # Check if we've just eaten a capsule
    if numCapsulesRemaining < self.numCapsulesRemaining:
      if numCarrying < 5:
        # Focus on getting food when we have capsule power
        foodListGoalsSorted = sorted(foodList, key=lambda goalNode: self.evaluateGoalNodeScore(gameState, goalNode, self.getMazeDistance(myPosition, (int(goalNode[0]), int(goalNode[1]))), "food"), reverse=True)
        for goalNode in foodListGoalsSorted:
          bestAction, totalCostOfOptimalPath = self.getAStarFirstMove(gameState, goalNode, "food", True)
          if totalCostOfOptimalPath < 9999:
            return bestAction 
      # Update capsule count
      self.numCapsulesRemaining = numCapsulesRemaining

    # If there are capsules, prioritize getting them
    numCapsulesRemaining = len(self.getCapsules(gameState))
    if numCapsulesRemaining > 0:
      capsules = self.getCapsules(gameState)
      if capsules:
        goalNode = capsules[0]
        bestAction, _ = self.getAStarFirstMove(gameState, goalNode, "capsule")
        if bestAction != "Stop":  # Only return if we found a valid path
          return bestAction 
    
    # If carrying little food, go for more food
    if gameState.getAgentState(self.index).numCarrying < 2:
      foodListGoalsSorted = sorted(foodList, key=lambda goalNode: self.evaluateGoalNodeScore(gameState, goalNode, self.getMazeDistance(myPosition, (int(goalNode[0]), int(goalNode[1]))), "food"), reverse=True)
      for goalNode in foodListGoalsSorted:
        bestAction, totalCostOfOptimalPath = self.getAStarFirstMove(gameState, goalNode, "food")
        if totalCostOfOptimalPath < 9999:
          return bestAction 
    else: 
      # Return food to home base
      borderListWithoutWalls = [borderPosition for borderPosition in borderList if borderPosition not in wallSet]
      if borderListWithoutWalls:  # Make sure we have valid border positions
        borderListSorted = sorted(borderListWithoutWalls, key=lambda goalNode: self.evaluateGoalNodeScore(gameState, goalNode, self.getMazeDistance(myPosition, (int(goalNode[0]), int(goalNode[1]))), "food"), reverse=True)
        for goalNode in borderListSorted:
          bestAction, totalCostOfOptimalPath = self.getAStarFirstMove(gameState, goalNode, "border")
          if totalCostOfOptimalPath < 9999:
            return bestAction 

    # Fall back to a simple behavior if all else fails
    actions = gameState.getLegalActions(self.index)
    return random.choice(actions)
  
  def evaluateGoalNodeScore(self, gameState, goalNodeCoords, numMovesToGoalNode, nodeType):
    features = self.getGoalFeatures(gameState, goalNodeCoords, numMovesToGoalNode, nodeType)
    weights = self.getGoalWeights()
    return features * weights

  def getGoalFeatures(self, gameState, goalNodeCoords, numMovesToGoalNode, nodeType):
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
  
  def registerInitialState(self, gameState):
    """
    Initialize the defensive agent and find good defensive positions
    """
    ReflexCaptureAgent.registerInitialState(self, gameState)
    
    # Calculate the middle of the board
    self.midX = gameState.data.layout.width // 2
    self.midY = gameState.data.layout.height // 2
    
    # Get good defensive positions near the middle line
    self.goodDefensivePositions = self.findGoodDefensivePositions(gameState)

  def findGoodDefensivePositions(self, gameState):
    """
    Find good defensive positions that are:
    1. On our side of the board
    2. Not walls
    3. Close to the middle line
    """
    width = gameState.data.layout.width
    height = gameState.data.layout.height
    wallGrid = gameState.getWalls()
    
    # Define the middle line based on team color
    if self.red:
        middleLine = self.midX - 1  # Red team defends left side
    else:
        middleLine = self.midX      # Blue team defends right side
    
    # Find positions close to the middle line that aren't walls
    goodPositions = []
    for y in range(1, height-1):  # Avoid the very top and bottom of the map
        if not wallGrid[middleLine][y]:
            goodPositions.append((middleLine, y))
    
    # If no good positions on the border, look one step inward
    if not goodPositions:
        inwardStep = -1 if self.red else 1
        newX = middleLine + inwardStep
        for y in range(1, height-1):
            if not wallGrid[newX][y]:
                goodPositions.append((newX, y))
    
    # Sort positions by centrality (closer to vertical center is better)
    goodPositions.sort(key=lambda pos: abs(pos[1] - self.midY))
    
    return goodPositions

  def getAStarFirstMove(self, gameState, goalNodeCoords):
    """
    Use A* search to find the first move toward the goal node
    """
    width, height = gameState.data.layout.width, gameState.data.layout.height
    wallGrid = gameState.getWalls()
    
    # Convert goal coordinates to integers and check if valid
    goalNodeCoords = (int(goalNodeCoords[0]), int(goalNodeCoords[1]))
    
    # Check if goal is valid
    if (goalNodeCoords[0] < 0 or goalNodeCoords[0] >= width or 
        goalNodeCoords[1] < 0 or goalNodeCoords[1] >= height or
        wallGrid[goalNodeCoords[0]][goalNodeCoords[1]]):
        # Return a safe default action
        return self.chooseSafeAction(gameState)
    
    minHeap = []
    uniquenessCounter = 1
    heapq.heappush(minHeap, (0, uniquenessCounter, (0, gameState, [])))
    
    while len(minHeap) > 0: 
      pop_res = heapq.heappop(minHeap)
      _, _, (g_n_curr, currState, listOfMoves) = pop_res
      
      currStateCoords = currState.getAgentState(self.index).getPosition()
      currStateCoords = (int(currStateCoords[0]), int(currStateCoords[1]))
      
      if currStateCoords == goalNodeCoords or uniquenessCounter > 400:
        if uniquenessCounter == 1:
          return "Stop"
        return listOfMoves[0][0] if listOfMoves else "Stop"

      legalActions = currState.getLegalActions(self.index)
      for legalAction in legalActions:
        # Skip "Stop" action to improve efficiency when there are other options
        if legalAction == "Stop" and len(legalActions) > 1:
            continue
            
        try:
            currSuccessorState = self.getSuccessor(currState, legalAction)
            successorCoords = currSuccessorState.getAgentState(self.index).getPosition()
            successorCoords = (int(successorCoords[0]), int(successorCoords[1]))
            
            h_n = self.getMazeDistance(successorCoords, goalNodeCoords)
            g_n = g_n_curr + 1
            uniquenessCounter += 1

            # Don't go to enemy territory (with a margin of safety)
            if (self.red and successorCoords[0] > width // 2) or (not self.red and successorCoords[0] < width // 2):
              g_n = 9999 + uniquenessCounter

            heapq.heappush(minHeap, (h_n + g_n, uniquenessCounter, (g_n, currSuccessorState, listOfMoves.copy() + [(legalAction, currSuccessorState)])))
        except Exception as e:
            # Skip this action if there's an error
            continue
    
    # If no path found, choose a safe action
    return self.chooseSafeAction(gameState)
  
  def chooseSafeAction(self, gameState):
    """
    Choose a safe action if pathfinding fails
    """
    actions = gameState.getLegalActions(self.index)
    bestAction = "Stop"
    bestScore = float("-inf")
    
    for action in actions:
        try:
            successor = self.getSuccessor(gameState, action)
            pos = successor.getAgentState(self.index).getPosition()
            pos = (int(pos[0]), int(pos[1]))
            
            # Don't go to enemy territory
            if (self.red and pos[0] >= self.midX) or (not self.red and pos[0] < self.midX):
                continue
                
            # Score each action based on distance to the middle
            score = -self.getMazeDistance(pos, (self.midX-1 if self.red else self.midX, self.midY))
            if score > bestScore:
                bestScore = score
                bestAction = action
        except Exception:
            continue
            
    return bestAction

  def findBestDefensivePosition(self, gameState, enemyPositions):
    """
    Find the best defensive position considering:
    1. Enemy locations (if visible)
    2. Food we're defending
    3. Default patrol positions along the middle line
    """
    width = gameState.data.layout.width
    height = gameState.data.layout.height
    
    # If enemies are visible, position to intercept them
    visibleEnemies = []
    for pos in enemyPositions:
        if pos is not None:
            # Make sure positions are integers and within bounds
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < width and 0 <= y < height and not gameState.hasWall(x, y):
                visibleEnemies.append((x, y))
    
    if visibleEnemies:
        # Get the closest enemy on our side or closest to our side
        closestEnemy = min(visibleEnemies, 
                          key=lambda pos: pos[0] if self.red else width - pos[0])
        return closestEnemy
    
    # No enemies visible, patrol the middle line at strategic positions
    defendingFood = self.getFoodYouAreDefending(gameState).asList()
    
    # If we have food to defend, pick a defensive position that's central to our food
    if defendingFood:
        try:
            # Find average position of our food
            avgX = sum(x for x, y in defendingFood) / len(defendingFood)
            avgY = sum(y for x, y in defendingFood) / len(defendingFood)
            
            # Find the defensive position closest to the average food position
            return min(self.goodDefensivePositions, 
                      key=lambda pos: self.getMazeDistance(pos, (int(avgX), int(avgY))))
        except Exception:
            # Fall back to a default position if there's an error
            pass
    
    # Default to a central defensive position
    if self.goodDefensivePositions:
        return self.goodDefensivePositions[0]
    else:
        # Create a valid position if none exists
        middleX = self.midX - 1 if self.red else self.midX
        for y in range(height):
            if not gameState.hasWall(middleX, y):
                return (middleX, y)
        # Last resort - find any legal position on our side
        for x in range(width):
            for y in range(height):
                if ((self.red and x < self.midX) or (not self.red and x >= self.midX)) and not gameState.hasWall(x, y):
                    return (x, y)
        # Absolute last resort
        return (self.midX - 1 if self.red else self.midX, self.midY)

  def chooseAction(self, gameState):
    """
    Choose the best defensive action
    """
    # Get enemy positions if visible
    opps = self.getOpponents(gameState)
    enemyPositions = []
    for i in opps:
        pos = gameState.getAgentPosition(i)
        if pos is not None:
            enemyPositions.append(pos)
        else:
            enemyPositions.append(None)
    
    try:
        # Find the best position to defend
        bestDefensivePosition = self.findBestDefensivePosition(gameState, enemyPositions)
        
        # Get the first move toward that position
        bestAction = self.getAStarFirstMove(gameState, bestDefensivePosition)
        
        # If pathfinding failed, fall back to a simple safe action
        if bestAction is None:
            bestAction = self.chooseSafeAction(gameState)
    except Exception:
        # Fall back to safe action if anything fails
        bestAction = self.chooseSafeAction(gameState)
    
    return bestAction