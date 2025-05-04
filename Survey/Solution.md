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

### Principles and Requirements for the Diffusion Process

 $q(v_T\mid v_0)=\mathrm{Cat}(v_0,\bar C_T)$
