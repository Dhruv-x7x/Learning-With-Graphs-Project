# Learning-With-Graphs-Project
This repository contains an implementation of the paper, "GRAPH-CONSTRAINED DIFFUSION FOR END-TO-END  PATH PLANNING" by Dingyuan Shi et. al. which was completed as a part of CS768 Learning With Graphs course at IIT Bombay in Spring 2025.

---

### Paper
In ```./Survey/``` directory you will find our notes that discuss the paper at length as well as the paper itself. The authors' codebase, which was used as a reference as well, is given [here](https://github.com/dingyuan-shi/Graph-Diffusion-Planning).

---

### Implementation

```research_2.ipynb``` contains a naive implementation of the GDP model. 

Note: Some functions are directly used from the authors' [own](https://github.com/dingyuan-shi/Graph-Diffusion-Planning/tree/main) implementation. We write our own forward process and diffusion process functions as well as the training functions. We perform our evaluation on synthetic maps generateed by ```nx.grid_2d_graph``` of 10x10 node graphs with random detours/edges. We generate 1000 paths with a path limitation of 30. The final sampling is done via **beam search** which is also used by the authors. We use a beam width of 50 for better metrics. The reproduction is faithful to the paper but does not fully achieve the same results due to computational constraints, not working with real world data and not tuning hyperparameters. 

We achieve the following metrics: ```Hit ratio: 0.75, Avg LCS: 2.10, Avg DTW: 15.30```
