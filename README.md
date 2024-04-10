# Vehicle Control

Implementation of some control algorithms, based on model-predictive control, for racing cars on a track. cascaded_mpc is based on [[1]](#1).

## Project Structure

- config: Contains configuration files encoded in a yaml format for different controllers, models and environments (track or trajectory). 
- controllers: Python modules related to controllers.
  - feedback_linearization: Module containing feedback linearization controllers.
  - mpc: Module containing model predictive controllers.
- environment: Python modules related to code for constructing the track and some trajectories.
- models: Classes representing different models (dynamic, kinematic, etc.)
- simulation: Python modules defining the simulation cycle 
- thirdparty: Third-party libraries (for now hsl libraries which deliver fast linear solvers for ipopt)

## Dependancies

- coinhsl (open thirdparty/coinhsl for instructions)
- packages listed in environment.yaml (you can install them with conda)

## Results

![alt](simulation/videos/cascaded_ippodromo.gif)

## References

<a id="1">[1]</a> 
[V. A. Laurense and J. C. Gerdes, "Long-Horizon Vehicle Motion Planning and Control Through Serially Cascaded Model Complexity," in IEEE Transactions on Control Systems Technology, vol. 30, no. 1, pp. 166-179, Jan. 2022](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9366415)
