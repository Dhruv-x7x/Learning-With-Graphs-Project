##  GRAPH-CONSTRAINED DIFFUSION FOR END-TO-END PATH PLANNING

Let's discuss the solution to the ![problem](/Survey/Problem.md). 

---

### How the Solution Works:

The solution to this problem involves two main parts:

1. **Unconditional Path Probability (Path Generation)**:
   - The authors use a **diffusion model** to learn the **distribution of paths**. This part of the process generates paths **unconditionally** (without any specific origin and destination). It allows the model to learn general patterns and structures of paths from the dataset.
   - After learning these patterns, the model can generate new paths by sampling from this distribution, even without specific destination or origin constraints.

2. **Conditional Sampling (Path Planning with Origin-Destination Pairs)**:
   - Once the model can generate paths unconditionally, the next step is to **condition** this generation on a specific **origin and destination pair**.
   - This is where the **prior evidence** $h(x|\text{ori}, \text{dst})$ comes into play. It encodes the information about the **origin** and **destination** and guides the model to generate a path that starts at the origin and ends at the destination.

---

### Why Diffusion?

Sequence-to-sequence models are also used in path pattern mining but the author's opted for diffusion for the following two reasons:
- Diffusion models show superior performance in complex generation tasks
- Diffusion models are flexible in incorporating categorical constraints compared to seq-to-seq models

---

### Principles and Requirements for the Diffusion Process

The categorical distribution of a node is modeled as $q(v_t | v_{t-1}) = Cat(v_t | p = q(v_{t-1})Q_t)$. Where $Q_t$ is the transition probability matrix. The choice of $Q_t$ is taken as $\alpha_t I_{|V|} + \beta_t \frac{11^T}{|V|}$

