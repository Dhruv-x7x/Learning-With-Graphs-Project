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

The categorical distribution of a node is modeled as $q(v_t | v_{t-1}) = Cat(v_t | p = q(v_{t-1})Q_t)$. Where $Q_t$ is the transition probability matrix. The choice of $Q_t$ is taken as $\alpha_t I_{|V|} + \beta_t \frac{11^T}{|V|}$ where $\beta_t = 1 - \alpha_t$, both control the forward diffusion process. Essentially $Q_t$ perturbs the original vertex's probability and redistributes it across other vertices. As $t$ approaches infinity, the distribution of $q(v_t)$ becomes uniform. 

The paper has the following requirements for the diffusion model:

- Offers a closed form for forward process.
- Ensures a computationally feasible posterior $q(v_t | v_{t-1}, v_0)$.
- Makes $q(v_T)$ independent of $q(v_0)$ for uninformed sample generation.
- Exhibits locality for small $t$ values

For this, the authors used a heat conduction partial differential equation to model the diffusion process.

---

### Forward Diffusion Process

From the partial differential equations we get the transition probability matrix $C_t = e^{t(A - D)}$ where $A$ is the adjacency matrix and $D$ is the degree matrix.

Properties of $C_{\tau}$:
- $C_{\tau}$ is a symmetric matrix. So  $C_{\tau}^T =  C_{\tau}$
- $C_{\tau 1 + \tau 2} =  C_{\tau 1} C_{\tau 2}$
- $C_{\tau} = \frac{11^t}{|V|}$ as $\tau$ approaches infinity if graph is connected
- $C_{\tau}$ = I as $\tau$ approaches infinity
- Summation of each row and column of $C_{\tau}$ is 1 for all $\tau > 0$ if graph is connected

From this we can model the diffusion as  $q(v_t | v_{t-1}) = Cat(v_t | q(v_{t-1})C_{\tau})$
