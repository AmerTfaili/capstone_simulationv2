# 50-Intersection SUMO Benchmark

This benchmark extends the existing paper-inspired style into a larger scenario with:
- exactly 50 signalized internal intersections
- one dominant west-east central corridor
- one stronger upper distributor and one central north-south spine
- selective feeder columns, bottlenecks, and oblique shortcuts
- asymmetric time-varying demand with two strong peak periods
- no preselected controlled intersections

## Files
- `generated/benchmark.net.xml`: compiled SUMO network
- `generated/benchmark.rou.xml`: routed demand
- `benchmark.sumocfg`: runnable SUMO configuration
- `scenario_metadata.json`: topology metadata and the intended 10-node control budget without a chosen subset

## Notes
- The network is engineered to support later budgeted control selection, but it does not rank or hardcode the 10 controlled intersections.
- The geometry is staggered and only partially cross-connected, so it behaves like an asymmetric corridor network rather than a regular grid.
- The strongest movement is the central corridor, followed by the upper distributor and the central spine. Lower bands and diagonal connectors serve as feeders and alternate pressure paths.
