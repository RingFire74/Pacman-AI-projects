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
from game import Actions, Agent, Directions

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
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

    def evaluationFunction(self, currentGameState, action):
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
        # print("---------", newGhostStates)
        "*** YOUR CODE HERE ***"
        # distances = []
        # for food in newFood.asList():
        # distances = []
        # if newFood.asList():
        distances = sorted(newFood.asList(), key=lambda x: manhattanDistance(newPos, x))#[manhattanDistance(newPos, food) for food in newFood.asList()]
        ghostDistance = [manhattanDistance(newPos,ghostPos.configuration.pos) for ghostPos in newGhostStates]
        # return max(max(newScaredTimes),min(distances))
        # print("--------",random.randint(0,2))
        # print("-----",distances,min(min(ghostDistance),min(distances)))
        # print("------",min(distances),successorGameState.getScore(), min(ghostDistance))
        multiplier = 0.25
        decpts = 0
        pts = min(min(ghostDistance),4)
        if distances:
            l = min(len(distances),2)
            # print("------------- ",l, len(distances))
            decpts = (min([mazeDistance(newPos,x,successorGameState) for x in distances[:l]]))#+ random.random()
        if pts == 4:
            if action == 'Stop':
                pts -= 10
            if min(ghostDistance) > 6:
                multiplier = 1
                # if distances:
                #     l = min(len(distances),2)
                #     # print("------------- ",l, len(distances))
                #     pts -= (min([mazeDistance(newPos,x,successorGameState) for x in distances[:l]]))#+ random.random()
        return successorGameState.getScore()+pts-(multiplier*decpts)#+min(min(ghostDistance),4)

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()
def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    return len(genericSearch(gameState,util.Queue(),point1,point2))
def genericSearch(problem, fringe,startState,goalState):

    visited = list()
    totalPath = list()
    fringe.push((startState, list(), 0))
    while not fringe.isEmpty():
        currentState = fringe.pop()
        # print("-------------------------------- print state ",currentState[0]) 
        if currentState[0] not in visited:
            if currentState[0] == goalState:
                return currentState[1]
            for childNode, action, childCost in getSuccessors(problem,currentState[0]):
                    totalPath = currentState[1].copy()
                    totalPath.append(action)
                    totalCost = currentState[2] + childCost
                    fringe.push((childNode, totalPath, totalCost))
            visited.append(currentState[0])
    return None

def getSuccessors(self, state):
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.getWalls()[nextx][nexty]:
                nextState = (nextx, nexty)
                successors.append( ( nextState, action, 1) )

        return successors

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

    def getAction(self, gameState):
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
        isPruning = False
        action = self.minMax(gameState, 0, 0, isPruning, float("-inf"), float("inf"))[1]

        return action
        util.raiseNotDefined()
    def minMax(self, gameState, index, depth, isPruning, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState),""
        
        if index == 0:
            return self.maxPacman(gameState, index, depth, isPruning, alpha, beta)
        
        else:
            return self.minGhost(gameState, index, depth, isPruning, alpha, beta)
        
    def maxPacman(self, gameState, index, depth, isPruning, alpha, beta):
        actions = gameState.getLegalActions(index)
        maxPts = float("-inf")
        maxAction = ""

        successorDepth = depth
        successorIndex = index + 1

        if successorIndex == gameState.getNumAgents():
            successorDepth += 1
            successorIndex = 0

        for action in actions:
            successor = gameState.generateSuccessor(index, action)

            pts = self.minMax(successor, successorIndex, successorDepth, isPruning, alpha, beta)[0]

            if pts >= maxPts:
                maxPts = pts
                maxAction = action

            if isPruning:
                if maxPts > beta:
                    return maxPts,maxAction
                alpha = max(alpha,maxPts)
        return maxPts, maxAction
    
    def minGhost(self, gameState, index, depth, isPruning, alpha, beta):
        actions = gameState.getLegalActions(index)
        minPts = float("inf")
        minAction = ""

        successorDepth = depth
        successorIndex = index + 1

        if successorIndex == gameState.getNumAgents():
            successorDepth += 1
            successorIndex = 0
        for action in actions:
            successor = gameState.generateSuccessor(index, action)
            
            pts = self.minMax(successor, successorIndex, successorDepth, isPruning, alpha, beta)[0]

            if pts < minPts:
                minPts = pts
                minAction = action

            if isPruning:
                if minPts < alpha:
                    return minPts,minAction
                beta = min(beta,minPts)
        return minPts, minAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        isPruning = True
        action = self.minMax(gameState, 0, 0, isPruning, float("-inf"), float("inf"))[1]

        return action
        util.raiseNotDefined()

    def minMax(self, gameState, index, depth, isPruning, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState),""
        
        if index == 0:
            return self.maxPacman(gameState, index, depth, isPruning, alpha, beta)
        
        else:
            return self.minGhost(gameState, index, depth, isPruning, alpha, beta)
        
    def maxPacman(self, gameState, index, depth, isPruning, alpha, beta):
        actions = gameState.getLegalActions(index)
        maxPts = float("-inf")
        maxAction = ""

        successorDepth = depth
        successorIndex = index + 1

        if successorIndex == gameState.getNumAgents():
            successorDepth += 1
            successorIndex = 0

        for action in actions:
            successor = gameState.generateSuccessor(index, action)

            pts = self.minMax(successor, successorIndex, successorDepth, isPruning, alpha, beta)[0]

            if pts >= maxPts:
                maxPts = pts
                maxAction = action

            if isPruning:
                if maxPts > beta:
                    return maxPts,maxAction
                alpha = max(alpha,maxPts)
        return maxPts, maxAction
    
    def minGhost(self, gameState, index, depth, isPruning, alpha, beta):
        actions = gameState.getLegalActions(index)
        minPts = float("inf")
        minAction = ""

        successorDepth = depth
        successorIndex = index + 1

        if successorIndex == gameState.getNumAgents():
            successorDepth += 1
            successorIndex = 0
        for action in actions:
            successor = gameState.generateSuccessor(index, action)
            
            pts = self.minMax(successor, successorIndex, successorDepth, isPruning, alpha, beta)[0]

            if pts < minPts:
                minPts = pts
                minAction = action

            if isPruning:
                if minPts < alpha:
                    return minPts,minAction
                beta = min(beta,minPts)
        return minPts, minAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    successorGameState = currentGameState#.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    # print("---------", newGhostStates)
    "*** YOUR CODE HERE ***"
    # distances = []
    # for food in newFood.asList():
    # distances = []
    # if newFood.asList():
    distances = sorted(newFood.asList(), key=lambda x: manhattanDistance(newPos, x))#[manhattanDistance(newPos, food) for food in newFood.asList()]
    ghostDistance = [manhattanDistance(newPos,ghostPos.configuration.pos) for ghostPos in newGhostStates]
    # return max(max(newScaredTimes),min(distances))
    # print("--------",random.randint(0,2))
    # print("-----",distances,min(min(ghostDistance),min(distances)))
    # print("------",min(distances),successorGameState.getScore(), min(ghostDistance))
    multiplier = 0.5
    decpts = 0

    pts = min(min(ghostDistance),4)

    if distances:
        l = min(len(distances),2)
        decpts = (min([mazeDistance(newPos,x,successorGameState) for x in distances[:l]]))

    if pts == 4:
        if min(ghostDistance) > 6:
            multiplier = 1
            
    return successorGameState.getScore()+pts-(multiplier*decpts)
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
