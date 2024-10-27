# Pacman-AI-projects
AI Pacman Project
Overview
This project is part of my AI course and involves implementing various search and adversarial algorithms to guide Pacman through mazes. The project is designed to teach fundamental AI concepts such as search algorithms, decision-making in adversarial environments, and optimization techniques.

In this project, Pacman must navigate mazes, avoid ghosts, and maximize the score by eating food pellets. The AI techniques applied include:

Search Algorithms (e.g., BFS, DFS, A* Search)
Adversarial Search (e.g., Minimax, Alpha-Beta Pruning)
Reinforcement Learning (e.g., Q-learning)
Probabilistic Models (e.g., Hidden Markov Models for ghost tracking)
Project Structure
The project is broken down into several parts:

Search Algorithms:

Implement various search algorithms to solve the maze efficiently and guide Pacman to food pellets.
Algorithms implemented: Depth-First Search (DFS), Breadth-First Search (BFS), Uniform Cost Search, and A* Search.
Adversarial Search:

Implement adversarial search techniques to play Pacman optimally against ghosts.
Techniques include Minimax, Expectimax, and Alpha-Beta Pruning.
Reinforcement Learning:

Implement reinforcement learning algorithms to train Pacman to make decisions in real-time.
Methods include Q-learning and approximate Q-learning.
Tracking:

Use probabilistic models to track the position of ghosts based on noisy distance observations using a Hidden Markov Model (HMM).
Key Files
pacman.py: The main file that handles the game logic.
search.py: Contains implementations for the search algorithms.
searchAgents.py: Agents that control Pacman using search algorithms.
multiAgents.py: Agents that control Pacman using adversarial search algorithms.
reinforcement.py: Implements Q-learning and other RL algorithms.
ghostAgents.py: Contains logic for ghost behavior.
layout/: Contains layout files for different maze configurations.
graphicsDisplay.py: Handles the graphical display of the game.
