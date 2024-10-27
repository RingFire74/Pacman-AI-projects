# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


from game import Directions
import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for _ in range(self.iterations):
            values_1 = self.values.copy()
            for state in self.mdp.getStates():
                Q = [float('-inf')]

                if self.mdp.isTerminal(state):
                    values_1[state] = 0

                else:
                    for action in self.mdp.getPossibleActions(state):
                        Q.append(self.getQValue(state, action))

                    values_1[state] = max(Q)

            self.values = values_1

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        return sum([T *(self.mdp.getReward(state, action, state_) + (self.discount * self.values[state_])) for state_, T in self.mdp.getTransitionStatesAndProbs(state, action)])
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***" 
        return max(self.mdp.getPossibleActions(state), default=None, key=lambda x: self.getQValue(state,x))
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()

        # initialize value function to all 0 values.
        for state in states:
            self.values[state] = 0

        n = len(states)

        for i in range(self.iterations):
            state = states[i % n]

            if not self.mdp.isTerminal(state):
                self.values[state] = self.getQValue(state, self.getAction(state))

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
        self.runValueIteration()

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = {}
        states = self.mdp.getStates()
        fringe = util.PriorityQueue()

        for s in states:
            self.values[s],predecessors[s] = 0,self.getPredecessors(s)

        for s in states:
            terminal = self.mdp.isTerminal(s)

            if not terminal:
                fringe.push(s, -abs(self.values[s] - self.max_Qvalue(s)))

        for _ in range(self.iterations):

            if fringe.isEmpty():
                return None

            s = fringe.pop()
            self.values[s] = self.max_Qvalue(s)

            for p in predecessors[s]:
                diff = abs(self.values[p] - self.max_Qvalue(p))
                if diff > self.theta:
                    fringe.update(p, -diff)


    def max_Qvalue(self, state):
        return max([self.getQValue(state, a) for a in self.mdp.getPossibleActions(state)])

    def getPredecessors(self, state):
        states = self.mdp.getStates()
        predecessors = set()

        if not self.mdp.isTerminal(state):

            for p in states:
                
                if not self.mdp.isTerminal(p):

                    for action in self.mdp.getPossibleActions(p):

                        for state_1, T in self.mdp.getTransitionStatesAndProbs(p, action):
                            if (state_1 == state) and (T > 0):
                                predecessors.add(p)

        return predecessors

