# EscapeTheSensors
For a thoroughly researched paper on this model and algorithm, please visit the "
Escape_The_Sensors-Research_Project_Masters.pdf" paper in the 'main' section.

## Overview

This repository contains the implementation of a Sensor Evasion Algorithm. The algorithm is designed to find an optimal path between a start and end point while minimising the probability of detection by a set of sensors.

## Versions

- **Real Terrain Version**: `EscapeTheSensors-RealTerrain.py` allows you to upload real terrain data to model a path on.
- **Function Landscape Version**: `EscapeTheSensors-FunctionLandscape.py` uses a function to map the terrain.
- **2D Landscape Version**: `EscapeTheSensors-2D.py` operates on a 2D landscape.

## Dependencies

- Python 3.x
- NumPy
- Matplotlib
- simanneal

## Algorithm

The algorithm uses Simulated Annealing to find the optimal path. It considers the following factors:

- **Probability of Detection**: Calculated based on the distance from sensors.
- **Speed**: The algorithm tries to maintain an optimal speed throughout the path.
- **Terrain**: The algorithm considers the height of the terrain to calculate the line-of-sight.

## Usage

1. **Initialisation**: Set the start and end points, and the positions of the sensors.
2. **Run Simulated Annealer**: This will return the best state, which is the optimal path, and the best energy, which is the minimum detection probability.
3. **Visualisation**: The algorithm provides functions to visualise the path and the energy states.

## Functions

- `PropSens(dist, sig)`: Calculates the probability of being sensed based on distance.
- `ProbDetect(Sensors, Point)`: Calculates the probability of detection at a given point.
- `LineOfSightResistance(Position, Sensor)`: Calculates the line-of-sight resistance between a sensor and a point.
- `TotalT_ThreshCalc(Start, End, Min_Speed)`: Calculates the total time threshold.
- `RunAnnealer(Initial_temp)`: Runs the Simulated Annealing algorithm.
- `Visualise(Iter_Points, Iter_Speeds, Iter_E, slicing)`: Visualises the path and energy states.

## Author

This code was developed as part of a research project.

## License

This project is licensed under the MIT License.
