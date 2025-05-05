# Learning-With-Graphs-Project
This repository contains an implementation of the paper, "GRAPH-CONSTRAINED DIFFUSION FOR END-TO-END  PATH PLANNING" by Dingyuan Shi et. al. which was completed as a part of CS768 Learning With Graphs course at IIT Bombay in Spring 2025.

---

### Paper
In ```./Survey/``` directory you will find our notes that discuss the paper at length as well as the paper itself. The authors' codebase, which was used as a reference as well, is given [here](https://github.com/dingyuan-shi/Graph-Diffusion-Planning).

---

### Implementation

```research_2.ipynb``` contains a naive implementation of the GDP model. Run all cells to see the outputs.

You can also run the following bash commands to run the project.

```bash
git clone https://github.com/Dhruv-x7x/Learning-With-Graphs-Project.git
cd Learning-With-Graphs-Project
```

Followed by 
```bash
python main.py
```

If you want to change parameters, you can do so at the top of the main.py where all parameters are defined. An explanation of all parameters is given below:

```python
# DEFINE YOUR ROAD NETWORK DIMENSIONS. FORMS A NUM_COLS x NUM_ROWS GRID WITH 25 EXTRA RANDOM CONNECTIONS
NUM_ROWS = 10 
NUM_COLS= 10
EXTRA_EDGES = 25

# DEFINE DIFFUSION PARAMETERS. TIMESTEP VALUES, SMALL T FOR LESS NOISE, HIGH T FOR MORE NOISE. YOU DON'T NECESSARILY NEED TO CHANGE THIS
TVALUES = [0.1, 0.5, 1, 2, 5]

# DEFINE PATHS PARAMETERS
NUM_PATHS = 1000 # NUMBER OF PATHS
MAX_PATH_LEN = 30 # MAXIMUM LENGTH OF A PATH

# DEFINE BEAM SEARCH PARAMETERS. BEAM SEARCH IS USED TO DRAW PREDICTED PATH SAMPLES FROM THE FINAL DISTRIBUTION 
BEAM_WIDTH = 50 # KEEPS TRACK OF 50 PATHS TO SEE WHICH ONE OF THEM REACHES DESTINATION FIRST
BEAM_MAX_LEN = 30 # MAX LENGTH OF A PATH

# SEED
SEED = 42 # SEED VALUE FOR REPRODUCIBILITY
```

Note: Some functions are directly used from the authors' [own](https://github.com/dingyuan-shi/Graph-Diffusion-Planning/tree/main) implementation. We write our own forward process and diffusion process functions as well as the training functions. We perform our evaluation on synthetic maps generateed by ```nx.grid_2d_graph``` of 10x10 node graphs with random detours/edges. We generate 1000 paths with a path limitation of 30. The final sampling is done via **beam search** which is also used by the authors. We use a beam width of 50 for better metrics. The reproduction is faithful to the paper but does not fully achieve the same results due to computational constraints, not working with real world data and not tuning hyperparameters. 

We achieve the following metrics: ```Hit ratio: 0.90, Avg LCS: 2.40, Avg DTW: 9.95```

---

### Structure of the Repository

```plaintext
|-- Survey 
  |-- 1317_GRAPH_CONSTRAINED_DIFFUSI.pdf # the paper being implemented
  |-- Experiments.md # describes the design of the experiments conducted by the authors
  |-- Problem.md # desrcibes the problem the paper is trying to solve
  |-- Solution.md # describes the solution proposed by the paper
|-- results # contains pictures of plots
|-- src
  |-- operations.py # contains the definitions of functions being used in main.py as well as class instances.
|-- main.py # runs the training process, calls functions from operations.py
|-- README.md
|-- research_2.ipynb # contains the notebook with our research. Can also be used as a substitute for main.py if you want to visualize at every step.
```
