## Experimental Design and Evaluation Metrics

### Evaluation Metrics

- **Dynamic Time Wrapping (DTW)**: Smaller DTW is better. Measures the similarity between two sequences that may vary in speed or length.
- **Longest Common Subsequence (LCS)**: Large LCS is better. Finds the longest common ordered subsequence between two sequences. We want generated paths to follow the ground truth paths as much as possible.
- **Hit Ratio**: Closer to 100% is better. We want for the paths to reach the destination.

### Dataset

The authors used two real world datasets of city A and city B from [Didi Gaia](https://outreach.didichuxing.com/SimulationS/data.html#:~:text=Participants%20can%20download%20the%20public%20dataset%20in%20the,Chengdu%20data%20set%20from%20the%20DiDi%20GAIA%20program.). 

### Baselines

The following algorithms were used to set baselines:
- Dijkstra's Algorithm (DA)
- NMLR
- Key Segment (KS)
- Navi from Amap
- CSSRNN
- MTNet
- HMM
- N-gram

### Results

With DA as baseline, GDP shows an 80% increase in LCS and a 30% decrease in DTW which outperforms all other algorithms including the industry standard Navi from Amap.
