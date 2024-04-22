# Vehicle Control

Implementation of some control algorithms, based on model-predictive control, for racing cars on a track.  
Cascaded MPC is an unofficial implementation of [[1]](#1).

## Install

To execute the code you need to install the following packages
- [miniconda](https://docs.anaconda.com/free/miniconda/index.html#quick-command-line-install)
- [coinhsl](https://github.com/neverorfrog/vehicle-control/tree/main/thirdparty/coinhsl#thirdparty-hsl)
- Create a conda environment with the required packages executing from the project root directory the command  
```conda env create --name vehicle-control -f environment.yaml``` 

## Project Structure

- config: Contains configuration files encoded in a yaml format for different controllers, models and environments (track or trajectory). 
- controllers: Python modules related to controllers.
  - feedback_linearization: Module containing feedback linearization controllers.
  - mpc: Module containing model predictive controllers.
- environment: Python modules related to code for constructing the track and some trajectories.
- models: Classes representing different models (dynamic, kinematic, etc.)
- simulation: Python modules defining the simulation cycle 
- thirdparty: Third-party libraries (for now hsl libraries which deliver fast linear solvers for ipopt)

## Results

### Cascaded MPC
![alt](simulation/videos/cascaded_ippodromo.gif)

### Singletrack MPC
![alt](simulation/videos/singletrack_ippodromo.gif)

### Cascaded MPC with obstacles
![alt](simulation/videos/cascaded_ippodromo_obstacles.gif)

## References

<a id="1">[1]</a> 
[V. A. Laurense and J. C. Gerdes, "Long-Horizon Vehicle Motion Planning and Control Through Serially Cascaded Model Complexity," in IEEE Transactions on Control Systems Technology, vol. 30, no. 1, pp. 166-179, Jan. 2022](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9366415)
