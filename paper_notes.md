# Notes: Network-wide traffic signal control based on the discovery of critical nodes and deep reinforcement learning

## Citation
Ming Xu, Jianping Wu, Ling Huang, Rui Zhou, Tian Wang, and Dongmei Hu. "Network-wide traffic signal control based on the discovery of critical nodes and deep reinforcement learning." Journal of Intelligent Transportation Systems, published online January 3, 2019. DOI: 10.1080/15472450.2018.1527694.

## Core idea
The paper argues that city-scale traffic signal control becomes hard because:
- the traffic state space is large
- the joint action space over many intersections grows too fast

Their solution is a two-stage framework:
1. Rank intersections and identify only the most critical nodes in the network.
2. Learn signal policies only for those critical nodes with a deep recurrent Q-network (DRQN).

This reduces the control problem from "optimize the whole network jointly" to "optimize a small set of high-impact intersections."

## Stage 1: Critical-node discovery
The paper uses CRRank, a trajectory-driven ranking method.

CRRank builds a tripartite graph with:
- OD pairs
- paths
- intersections

It then propagates scores across this graph to rank intersection importance. The ranking is meant to combine:
- traffic flow / capacity
- centrality
- path irreplaceability or substitutability
- upstream/downstream spatial influence among neighboring nodes

The paper positions this as better than using only flow, only betweenness, or random selection.

## Stage 2: Signal control with DRQN
Each selected critical intersection is controlled independently by a DRQN.

### Why DRQN instead of plain DQN
The paper argues traffic is partially observable and strongly temporal. A single observation does not describe the evolving queue pattern well enough, so they add an LSTM on top of Q-learning.

### State
Each agent observes:
- vehicle counts on lanes of the controlled intersection
- average speeds on those lanes
- vehicle counts on lanes of neighboring intersections
- average speeds on those lanes
- current signal-state information for the controlled intersection and neighbors

### Action
The controller does not choose arbitrary phase sets. It only chooses between:
- `N`: keep the current phase
- `A`: advance to the next phase in a fixed loop

The phase loop is:
- East-West Green
- East-West Left-turn Green
- North-South Green
- North-South Left-turn Green

A 3-second yellow transition is inserted automatically and is not learned as a separate action.

This is a deliberate simplification to keep the action space tractable and safe.

### Reward
The paper says average delay alone can bias the system toward major flows and starve minor approaches. To reduce that effect, it uses a congestion-style reward shaped by waiting time, with a tolerable waiting-time threshold `C = 60` and constants `g = 0.15`, `q = 2`.

The design intent is fairness-aware queue control, not just average-delay minimization.

## Experimental setup
- Simulator: SUMO
- Learning stack: TensorFlow
- Synthetic networks: 20, 50, and 100 nodes
- Episode length: 3600 simulation steps
- Demand process: Poisson arrivals
- Controlled intersections: top 10 critical nodes from CRRank

Baselines for node selection:
- random choice
- flow-based ranking
- betweenness-based ranking
- CRRank

Baselines for control:
- fixed signal timing
- self-organizing traffic lights (SOTL)
- tabular Q-learning
- DQN
- DRQN

## Main results
### Critical-node selection
Applying DRQN to nodes chosen by CRRank produced lower travel time than applying it to nodes chosen by random, flow-only, or betweenness-only ranking, especially in the larger 50-node and 100-node networks.

### Control algorithm
DRQN outperformed fixed timing, SOTL, tabular Q-learning, and DQN after convergence.

Reported average travel time after convergence:

| Network | FST | SOTL | QL | DQN | DRQN |
|---|---:|---:|---:|---:|---:|
| 20 nodes | 196.3 | 166.4 | 173.8 | 146.8 | 117.5 |
| 50 nodes | 304.7 | 286.2 | 278.9 | 263.8 | 221.6 |
| 100 nodes | 497.1 | 476.3 | 448.0 | 412.5 | 379.8 |

The paper also reports that its reward design lowers maximum delay compared with a simpler delay-based reward, which is meant to reduce extreme waiting times for minority traffic streams.

## What the paper contributes
- A decomposition strategy for large-scale traffic signal control
- A data-driven node-ranking method that uses trajectories rather than pure topology
- A recurrent RL controller for temporal traffic dynamics
- An argument that controlling only a few strategically chosen intersections can yield network-wide benefit

## Important limitations
1. The paper is not full network-wide joint control.
Only the top critical nodes are learned; the rest remain fixed-timing. The title is broader than the actual control scope.

2. The experiments are synthetic.
The trajectory data used in evaluation are generated from SUMO rather than collected from a real city deployment.

3. Agent coordination is limited.
Each critical node is controlled independently, even though neighbor observations are included. There is no learned multi-agent coordination mechanism.

4. Baselines are relatively weak by modern standards.
The comparisons do not include stronger later methods such as graph-based MARL, transformer-based policies, PPO variants, or value decomposition methods.

5. Action space is heavily constrained.
Choosing only between "hold" and "advance" makes learning easier, but it also limits the controller's flexibility.

6. Scalability claim is partly achieved by problem reduction.
The method scales by shrinking the set of controlled nodes, not by solving the full joint optimization problem directly.

## Practical reading of the paper
The strongest idea in the paper is not the DRQN by itself. The real contribution is the combination:
- first identify high-leverage intersections
- then spend learning capacity only where it matters most

If you are using this paper for a project, that is the main takeaway worth carrying forward.

## How to position this paper
This paper is useful if you want to justify:
- importance-based selection of intersections before RL control
- temporal models such as LSTM for signal control
- a pragmatic city-scale approach when full multi-agent optimization is too expensive

It is less useful as evidence for:
- fully decentralized cooperation
- real-world deployment readiness
- state-of-the-art RL performance today
