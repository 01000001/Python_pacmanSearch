# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def generic_search(problem, fringe, add_to_fringe_fn):
    closed = set()
    start = (problem.getStartState(), 0, [])  # (node, cost, path)
    add_to_fringe_fn(fringe, start, 0)

    while not fringe.isEmpty():
        (node, cost, path) = fringe.pop()

        if problem.isGoalState(node):
            return path

        if not node in closed:
            closed.add(node)

            for child_node, child_action, child_cost in problem.getSuccessors(node):
                new_cost = cost + child_cost
                new_path = path + [child_action]
                new_state = (child_node, new_cost, new_path)
                add_to_fringe_fn(fringe, new_state, new_cost)

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"

    #### TRY REAL IMPLEMENTATION

    from util import Stack

    fringe = Stack()
    
    def add_to_fringe_fn(fringe, node, cost):
        fringe.push(node)

    return generic_search(problem, fringe, add_to_fringe_fn)

    """
    # implementing this pseudocode: http://bit.ly/1LqdHyY
    
    # import stack from util.py
    from util import Stack

    # closed <= an empty set
    closed = []

    # fringe <- INSERT(MAKE-NODE(INITIAL-STATE[problem]), fringe)
    fringe = Stack()
    fringe.push([[problem.getStartState()],[]])
    

    # loop do
    while True:

        # if fringe is empty then return failure
        if fringe.isEmpty():
            return []

        # node <- REMOVE-FRONT(fringe, strategy)
        node = fringe.pop()

        # if GOAL-TEST(problem, STATE[node]) then return node
        if problem.isGoalState(node[0][-1]):
            return node[1]

        # if STATE[node] is not in closed then
        if node[0][-1] not in closed:

            # add STATE[node] to closed
            closed.append(node[0][-1])

            # for child-node in EXPAND(STATE[node], problem) do
            
            for child_node in problem.getSuccessors(node[0][-1]):

                # fringe <- INSERT(child-node, fringe)
                fringe.push([node[0] + [child_node[0]], node[1]+[child_node[1]]])
               
                
    #comment out rainisng the exception
    #util.raiseNotDefined()
    """
    """
    ### V2 sem az enyime ok
    fringe = util.Stack()
    current_state = [problem.getStartState(), []]
    successors = None
    visited_states = set()
    item = None
    while(not problem.isGoalState(current_state[0])):
        (current_pos, directions) = current_state 
        successors = problem.getSuccessors(current_pos)
        for item in successors:
            fringe.push((item[0], directions + [item[1]]) )
        while (True):
            if (fringe.isEmpty()):
                return None
            item = fringe.pop() 
            if (item[0] not in visited_states):
                break    
        print item
        current_state = item
        visited_states.add(item[0])
    # print current_state[1]
    return current_state[1]
    # util.raiseNotDefined()
    """

    """
    ### V3 Yet another one pass
    closed = []
    fringe = util.Stack()
    fringe.push([[problem.getStartState()],[]])
    while True:
        if fringe.isEmpty():
            return []
        node = fringe.pop()
        if problem.isGoalState(node[0][-1]):
            return node[1]
        if node[0][-1] not in closed:
            closed.append(node[0][-1])
            for child_node in problem.getSuccessors(node[0][-1]):
                fringe.push([node[0] + [child_node[0]], node[1]+[child_node[1]]] )
    util.raiseNotDefined()
    """


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    from util import Queue

    fringe = Queue()
    
    def add_to_fringe_fn(fringe, node, cost):
        fringe.push(node)

    return generic_search(problem, fringe, add_to_fringe_fn)

    """
    # implementing this pseudocode: http://bit.ly/1LqdHyY modified to a Queue
    # see here http://en.wikipedia.org/wiki/Breadth-first_search#Pseudocode
    
    # import stack from util.py
    from util import Queue

    # closed <= an empty set
    closed = []

    # let fringe be a queue
    # fringe <- INSERT(MAKE-NODE(INITIAL-STATE[problem]), fringe)
    fringe = Queue()
    fringe.push([[problem.getStartState()],[]])
    

    # loop do
    while True:

        # if fringe is empty then return failure
        if fringe.isEmpty():
            return []

        # node <- REMOVE-FRONT(fringe, strategy)
        node = fringe.pop()

        # if GOAL-TEST(problem, STATE[node]) then return node
        if problem.isGoalState(node[0][-1]):
            return node[1]

        # if STATE[node] is not in closed then
        if node[0][-1] not in closed:

            # add STATE[node] to closed
            closed.append(node[0][-1])

            # for child-node in EXPAND(STATE[node], problem) do
            
            for child_node in problem.getSuccessors(node[0][-1]):

                # fringe <- INSERT(child-node, fringe)
                fringe.push([node[0] + [child_node[0]], node[1]+[child_node[1]]])
               """
    #util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue

    fringe = PriorityQueue()
    
    def add_to_fringe_fn(fringe, node, cost):
        fringe.push(node, cost)

    #print "Start:", problem.getStartState()
    #print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    #print "Start's successors:", problem.getSuccessors(problem.getStartState())


    return generic_search(problem, fringe, add_to_fringe_fn)

    #util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def cost_plus_heuristic(item):
    return item[2] + item[3]

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"

    startNode = problem.getStartState()
    fringe = util.PriorityQueueWithFunction(cost_plus_heuristic)
    start = (startNode, [], 0, heuristic(startNode, problem))
    
    fringe.push(start)

    visited= {startNode: cost_plus_heuristic(start)}
    
    expanded_nodes = {}

    while not fringe.isEmpty():

        current_item = fringe.pop()
        current_node, currentPath, currentCost, currentHeuristic = current_item

        if current_node not in expanded_nodes.keys():

            expanded_nodes[current_node] = cost_plus_heuristic(current_item)
            
            if problem.isGoalState(current_node):
                return currentPath
            else:
                for tmp in problem.getSuccessors(current_node):

                    node, action, cost = tmp

                    item = (node, currentPath + [action], currentCost + cost, heuristic(node, problem))
                    
                    if node in visited.keys():
                        if visited[node] > cost_plus_heuristic(item) :
                            fringe.push(item)
                            visited[node] = cost_plus_heuristic(item)
                    else:
                        fringe.push(item)
                        visited[node] = cost_plus_heuristic(item)

    
    #util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
