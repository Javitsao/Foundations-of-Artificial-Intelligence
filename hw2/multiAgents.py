# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        min_food = float('inf')
        min_ghost = float('inf')
        
        for food in newFood.asList():
            dist = manhattanDistance(food, newPos)
            if dist < min_food:
                min_food = dist
        
        for ghoststate in newGhostStates:
            ghostpos = ghoststate.getPosition()
            dist = manhattanDistance(ghostpos, newPos)
            if dist < min_ghost and dist != 0:
                min_ghost = dist
        
        return successorGameState.getScore() + 7/min_food - 3/min_ghost

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    
    
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def mini(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            v = float('inf')
            if agentIndex == gameState.getNumAgents() - 1:
                for action in gameState.getLegalActions(agentIndex):
                    v = min(v, maxi(agentIndex, depth - 1, gameState.generateSuccessor(agentIndex, action)))
            else:
                for action in gameState.getLegalActions(agentIndex):
                    v = min(v, mini(agentIndex + 1, depth, gameState.generateSuccessor(agentIndex, action)))
            return v
        
        def maxi(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            v = float('-inf')
            for action in gameState.getLegalActions(0):
                v = max(v, mini(1, depth , gameState.generateSuccessor(0, action)))
            return v
        #return (maxi(gameState.getNumAgents(), self.depth, gameState))
        actions = gameState.getLegalActions(0)
        best = ""
        v = -float('inf')
        for action in actions:
            nextState = gameState.generateSuccessor(0, action)
            score = mini(1, self.depth, nextState)
            if score > v:
                v = score
                best = action
        return best
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def mini(agentIndex, depth, gameState, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            v = float('inf')
            actions = gameState.getLegalActions(agentIndex)
            if agentIndex == gameState.getNumAgents() - 1:
                for action in actions:
                    v = min(v, maxi(agentIndex, depth - 1, gameState.generateSuccessor(agentIndex, action), alpha, beta))
                    if v < alpha:
                        return v
                    beta = min(beta, v)
            else:
                for action in actions:
                    v = min(v, mini(agentIndex + 1, depth, gameState.generateSuccessor(agentIndex, action), alpha, beta))
                    if v < alpha:
                        return v
                    beta = min(beta, v)
            return v
        
        def maxi(agentIndex, depth, gameState, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            v = float('-inf')
            actions = gameState.getLegalActions(0)
            for action in actions:
                v = max(v, mini(1, depth, gameState.generateSuccessor(0, action), alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v
        
        actions = gameState.getLegalActions(0)
        best = ""
        v = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        for action in actions:
            nextState = gameState.generateSuccessor(0, action)
            score = mini(1, self.depth, nextState, alpha, beta)
            alpha = max(alpha, score)
            if score > v:
                v = score
                best = action
        return best
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            numAgents = gameState.getNumAgents()
            if agentIndex >= numAgents:
                return expectimax(0, depth - 1, gameState)

            if agentIndex == 0:
                return max(expectimax(agentIndex + 1, depth, gameState.generateSuccessor(agentIndex, action)) for action in gameState.getLegalActions(agentIndex))

            else:
                ghostIndex = agentIndex % numAgents
                actions = gameState.getLegalActions(ghostIndex)
                numActions = len(actions)
                return sum(expectimax(agentIndex + 1, depth, gameState.generateSuccessor(ghostIndex, action)) for action in actions) / numActions

        return max(gameState.getLegalActions(0), key=lambda action: expectimax(1, self.depth, gameState.generateSuccessor(0, action)))

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    capsules = currentGameState.getCapsules()  # Add this line
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    minFoodDist = float('inf')
    minGhostDist = float('inf')
    minCapsuleDist = float('inf')  # Add this line
    for foodPos in food.asList():
        dist = manhattanDistance(foodPos, pacmanPos)
        if dist < minFoodDist:
            minFoodDist = dist
    for capsule in capsules:
        dist = manhattanDistance(capsule, pacmanPos)
        #print(capsule, pacmanPos, dist)
        if dist < minCapsuleDist:
            minCapsuleDist = dist
    for ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        dist = manhattanDistance(ghostPos, pacmanPos)
        if dist < minGhostDist and dist != 0:
            minGhostDist = dist
    #print(currentGameState.getScore() + 3 / minFoodDist + 30 / minCapsuleDist - 2 / minGhostDist, currentGameState.getScore(), 3 / minFoodDist, 30 / minCapsuleDist, 2 / minGhostDist)
    return currentGameState.getScore() + 5 / minFoodDist + 87 / minCapsuleDist - 3 / minGhostDist

# Abbreviation
better = betterEvaluationFunction