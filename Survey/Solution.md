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

---

### Reverse Process

The reverse process is modeled as,

$$
q(v_{t-1} | v_t, \hat v_0) = Cat(v_{t-1} | p \propto v_tC_t \circ \hat v_0 \bar C_{t-1})
$$

During the reverse process we do not have access to the original vertex $v_0$ and that makes sense because we don't want our model to cheat and use the ground truth to find the ground truth. We make assumptions and use an estimated value of $\hat v_0$. The categorical distribution combines information from $v_tC_t$, which is how we got from $v_0$ to $v_T$, and $\hat v_0 \bar C_{t-1}$, which is how we are supposed to go from $v_{t-1}$ back to the original vertex. Derivations in Appendix B.

---

### Diffusion Process for a Path

Thereare two ways in which paths can be diffused: the first one involves maintaining vertex connectivity within the path while the other diffuses each vertex independently. The authors opt for option 2 for the following reasons:

- Preserving connectivity without any information loss during the diffusion process is a difficult task because of conditional probabilities like $p(x_t^i | x_t^{i-1})$ that are introduced. This directly conflicts with diffusion models' non auto-regressive nature. Auto-regressive means that the vertices have a temporal dependency with each other which makes things complicated. Diffusion models are non auto-regressive in nature for tractability and simplicity.
- Strict Connectivity makes $q(v_T)$ overly reliant on $q(v_0)$ which directly violates a requirement of the diffusion model as described above.

So we treat the diffusion as contextual, independently diffusing each vertex such that for small $t$ values the noisy path is not too far off from the original path and for high $t$ values it is completely random, aiding sample generation. 

---


