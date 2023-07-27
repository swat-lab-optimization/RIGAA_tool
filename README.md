
<!-- <p align="center">
	<img height="400px" src="img/output_robot.gif"/>
    <img height="400px" src="img/output_vehicle.gif"/>
</p> -->

<p float="center" align="center">
  <img src="img/output_robot.gif" width="350" />
  <img src="img/output_vehicle.gif" width="350" /> 
</p>

<p float="center" align="center">
  <img src="img/output_robot.gif" width="350" />
  <img src="img/gif_veh.gif" width="350" /> 
</p>

<h1 align="center">
	Reinforcement learning Informed Genetic Algorithm for Autonomous systems testing (RIGAA)
</h1>

<p align="center">
  <b>Current Tool Version: 0.1.0</b>
</p>

## Installation

### Installation instructions to run the ant agent in Mujoco simulator for autonomous robot case study
1. We could make it work on Ubuntu 20.04 LTS with python 3.8 or 3.9, with Pytroch with Cuda available. Running on Ubuntu virtual machine on Windows did not work.
2. Clone this repository to your machine. Then change the directory.

```
cd RIGAA_tool
```  
3. First install the dependencies needed to run the RIGAA tool
```
pip install -r requirements.txt
```  
3. Go to the d4rl folder.
```
cd d4rl
```
4. Install the dependencies for the d4rl project.
```
pip install -e .
```
5. In addition you also need to install pytorch library with cuda support (including torch and torchvision). The correct command to perform the installation can be found [here](https://pytorch.org/get-started/locally/).
6. You also need to set up the [mujoco simulator](https://github.com/openai/mujoco-py): download the MuJoCo version 2.1 binaries for
[Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) or
[OSX](https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz).
Extract the downloaded `mujoco210` directory into `~/.mujoco/mujoco210`.
7. Install the requirements for *RLkit* library:
```
cd rlkit-offline-rl-benchmark
pip install -e .
```
8. Add following line to .bashrc changing *your_path* accordingly:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/your_path/.mujoco/mujoco210/bin
```
8. Now almost everything is set-up.
