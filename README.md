# Vehicle Control

Implementation of some control algorithms, based on model-predictive control, for racing cars on a track.  
Cascaded MPC is an unofficial implementation of [[1]](#1).

## Install

### Coinhsl (taken from [ThirdParty-HSL](https://github.com/coin-or-tools/ThirdParty-HSL))
Linear solvers that make code significantly faster
1. Install dependencies  
```sudo apt install gcc g++ gfortran liblapack-dev libmetis-dev pkg-config --install-recommends```
2. Obtain a tarball with Coin-HSL source code from https://licences.stfc.ac.uk/product/coin-hsl
3. Unzip the tarball `coinhsl-x.y.z.tar.gz` into `thirdparty/coinhsl` and rename the unzipped folder to `coinhsl`
4. Run the following commands form inside `thirdparty/coinhsl` to install coinhsl libraries
```
./configure
make -j12
sudo make install
```
You should get a message like  
_Libraries have been installed in:  
   /usr/local/lib_


### Conda Environment
You can install the packages in environment.yaml one-by-one with pip, but I still suggest using virtual environments like conda or venv
1. Install [miniconda](https://docs.anaconda.com/free/miniconda/index.html#quick-command-line-install)
2. Create a conda environment with the required packages executing from the project root directory the command  
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

## Concept

The goal is to enhance racing performance by pushin the vehicle to its physical limits while achieving good computational performance. When using NMPC, computation times can become prohibitive with a longer planning horizon. The proposal of [[1]](#1) is to be able to plan far ahead in the future while keeping things sufficiently efficient. This entails also a better racing performance.

## Results

### Cascaded (green) vs Singletrack (yellow)
![alt](simulation/videos/race_ippodromo.gif)

### Cascaded vs Singletrack with obstacles
![alt](simulation/videos/race_obstacles_ippodromo.gif)


## References

<a id="1">[1]</a> 
[V. A. Laurense and J. C. Gerdes, "Long-Horizon Vehicle Motion Planning and Control Through Serially Cascaded Model Complexity," in IEEE Transactions on Control Systems Technology, vol. 30, no. 1, pp. 166-179, Jan. 2022](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9366415)

## Possible expansions

- Learning-based MPC
- Observers for the state
- More realistic track
