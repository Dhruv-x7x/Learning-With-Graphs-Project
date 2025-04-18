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

