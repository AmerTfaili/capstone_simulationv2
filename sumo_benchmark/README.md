# 20-Intersection SUMO Benchmark

This benchmark implements a paper-inspired nonuniform network with:
- exactly 20 signalized internal intersections
- one dominant west-east corridor
- one secondary north-south spine
- weaker feeders and outer links
- asymmetric time-varying demand with two strong peak periods

## Files
- `generated/benchmark.net.xml`: compiled SUMO network
- `generated/benchmark.rou.xml`: routed demand
- `benchmark.sumocfg`: runnable SUMO configuration
- `control_roles.json`: CRRank-style node ranking and the 10 DRQN-controlled intersections

## Selected DRQN Nodes
J08, J09, J10, J07, J06, J03, J13, J18, J14, J12

## Fixed-Time Nodes
J02, J20, J04, J11, J15, J19, J17, J05, J01, J16

## Notes
- The network is intentionally not a uniform open grid.
- The strongest links are the main corridor, then the central spine.
- The scenario is designed so poor signal decisions create visible congestion in the corridor core without forcing permanent gridlock.
