##  GRAPH-CONSTRAINED DIFFUSION FOR END-TO-END PATH PLANNING

### Abstract

Think of Google Maps, which usually finds the shortest or fastest route. But people don’t always take the most “optimal” route, they might prefer to avoid traffic, stick to familiar roads, or pass by their favorite chai shop. Traditional systems which involve explicit optimization, that is, using predefined rules to compute the shortest or fastest route, struggle to understand these hidden preferences. This paper introduces GDP, a data-driven path-planning model that learns from real-world travel patterns. Instead of rigidly following shortest-path algorithms, “it uses a novel diffusion process that incorporates constraints from road networks, and plans paths as conditional path generation given the origin and destination as prior evidence”. The paper boasts 14.2% ~ 43.5% improvement over strong baselines on real-world datasets. In our project we will attempt to reproduce results in this given range. 

---

### Introduction

Path-planning is an important topic to research about. It is used everywhere in transportation, logistics, emergency services, robotics, etc., Traditionally, we find the shortest or the most optimized path by minimizing a certain criteria like **time** or **distance**, or we have edge weights based on a combination of the two and other factors. We then use search-based algorithms like A* and Dijkstra's. While we do get the most optimized paths with very high probability, it hardly ever reflects real world datasets. User intentions are too complex to model in closed form, for example, people might not always want to take the shortest route, they might prefer a more scenic one instead. This paper bypasses the need for search based methods via a data-driven diffusion based approach. The idea for this diffusion was inspired by heat conduction diffusion. 

#### Linear Accumulative Cost Assumption

Real world paths don't always have linear costs. Which means that if the edge weights of a path are 2, 3, and 5 the total cost won't be 2+3+5 = 10. For example, an EV battery does not discharge linearly with distance, it may discharge more going uphill for example. This assumption is present in traditional path planing algorithms and leads to bias.

#### Other Learned Path Generation Methods

There exists other data-driven path generation models such as simple markov models, sequence-to-sequence neural networks. These methods sure are data-driven but also unconditional, which means that they can generate paths based on real-world data patterns but fail to give constrained paths given a Origin-Destination pair. This paper introduces GDP, which generated conditional paths give a OD pair.

---

### Graph Constrained Diffusion Model

The model starts with a noisy distribution or like a random graph and gradually refines this by 'spreading out' the probability over the constrained paths in the same way as how heat diffusion works in metals. This part is called the 'Forward pass'. This continues until we get a valid, structured path distribution. The entire process of generating a path is handled by GDP itself without needing extra optimization steps. 

The paper also "exploits a tailored self-attention mechanism" to keep in check the origin, destination and current path while generating paths. This ensures a spatio-temporal relation with the model at all times, and is used as prior evidence for the conditional sampling via node2vec.

#### How the training works

It gradually adds noise to real paths and adds vertex probabilities across the graph and then it learns to denoise these paths. Essentially it should then generate real paths from random initial states. They key innovation here is to use **Graph Laplacian** to constrain the diffusion process to valid paths.

---

### Problem Definitions

#### Definition 1 (Path)
A path $x$ on a road network is defined as a sequence of vertices $(v_0, v_1, ..., v_{|x|})$ where each pair of consecutive vertices are adjacent in the graph. Formally:
$$
x = (v_0, v_1, ..., v_{|x|}) \quad \text{where} \quad (v_i, v_{i+1}) \in E, \quad \forall i = 0, 1, ..., |x|-1
$$
where $(v_i, v_{i+1})$ denotes an edge between vertices $v_i$ and $v_{i+1}$.

#### Definition 2 (Path Planning)
Given a road network graph $G = \langle V, E \rangle$ with edges weighted by $w(v_i, v_j)$, path planning aims to find a path $x = (v_0, ..., v_{|x|})$ that minimizes the total path cost between an origin $\text{ori}$ and destination $\text{dst}$, defined as:

$$
\min \sum_{i=0}^{|x|-1} w(v_i, v_{i+1})
$$

where $w(v_i, v_{i+1})$ is the weight (cost) assigned to the edge connecting vertices $v_i$ and $v_{i+1}$.

#### Definition 3 (End-to-End Path Planning)
End-to-end path planning is formulated as the task of planning paths between a given origin $\text{ori}$ and destination $\text{dst}$ by generating paths that follow the distribution of real-world path data $P$, conditioned on the origin and destination:

$$
p_\theta(x) = p_\theta(x) \cdot h(x|\text{ori}, \text{dst})
$$

where $p_\theta(x)$ is the probability distribution of paths (learned from the dataset $P$), and $h(x|\text{ori}, \text{dst})$ represents the prior information conditioned on the origin and destination.

---

### Notations

1. **$G = \langle V, E \rangle$**:
   - A **graph** with vertices $V$ and edges $E$.
   - **$V$**: Set of **vertices** (locations or intersections).
   - **$E$**: Set of **edges** (roads between vertices).

2. **$x, P$**:
   - **$x$**: A **path** represented as a sequence of vertices $(v_0, v_1, ..., v_{|x|})$.
   - **$P$**: A **path dataset** containing multiple paths.

3. **$(v_0, v_1, ..., v_{|x|})$**:
   - A **sequence of vertices** forming a path $x$, where each consecutive pair $(v_i, v_{i+1})$ represents an edge in the graph.

4. **$\text{ori}, \text{dst}$**:
   - The **origin** and **destination** vertices in the path planning task.

5. **$x_i, v_i$**:
   - $x_i$: The **i-th vertex** in the path $x$.
   - $v_i$: The **i-th vertex** in the graph $G$.

6. **$x_t$**:
   - The **diffused path** at **time step $t$** in the diffusion process.

7. **$Q$**:
   - The **transition probability matrix** for the graph, representing the probabilities of transitioning from one vertex to another.

8. **$A, D$**:
   - **$A$**: The **adjacency matrix** of the graph, where $A[i, j] = 1$ if there is an edge between vertices $v_i$ and $v_j$, otherwise 0.
   - **$D$**: The **degree matrix**, where $D[i, i]$ represents the number of edges connected to vertex $v_i$.

9. **$M[i, j]$**:
    - The element at **row $i$** and **column $j$** of matrix $M$.

10. **$M[:, j] / M[i, :]$**:
    - **$M[:, j]$**: The **j-th column** of matrix $M$.
    - **$M[i, :]$**: The **i-th row** of matrix $M$.

11. **$C_\tau$**:
    - The **transition probability matrix** at time $\tau$ for the diffusion process.

12. **$p, q(\cdot)$**:
    - **$p$**: A **row vector** representing a categorical distribution.
    - **$q(\cdot)$**: A **row vector** representing a categorical distribution of moving between vertices.

13. **$v$**:
    - A **one-hot vector** representing a vertex. The one-hot encoding has 1 at the index of the vertex and 0 elsewhere.

14. **$\hat{v}$**:
    - The **estimated distribution** for a vertex after the diffusion process.

15. **Cat($\cdot | p$)**:
    - A **categorical random variable** sampled according to the probability distribution $p$.

---

