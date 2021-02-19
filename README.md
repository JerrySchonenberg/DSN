# Differential Siamese Network for the Avoidance of Moving Obstacles
### BSc Computer Science Thesis - Jerry Schonenberg
##### Leiden Institute of Advanced Computer Science (LIACS) - 26-08-2020
This repository provides the thesis, code and results from my BSc Project. The project tackles the problem of avoiding moving obstacles using only a monocular RGB camera. It uses a convolutional neural network (CNN) to retrieve information from the images, where its [architecture](/models/baseline/original/baseline_paper.py) is taken from another paper from Khan and Parker ("_Vision based indoor obstacle avoidance using a deep convolutional neural network_"). The architecture is [modified](/models/baseline/modified/baseline_modified.py) slightly and acts as a baseline for our solution to the problem. Its architecture is visualized below.
![Architecture](/models/baseline/modified/img/visualization.png)
This README contains a brief overview of the project and how to run important code. For a more thorough explanation, please read the [thesis](thesis/thesis.pdf). All experiments are conducted in a virtual environment, we chose for [CoppeliaSim](https://www.coppeliarobotics.com/) and this software is also required if you want to run any of the code.

___

### Differential Siamese Network
The Differential Siamese Network (DSN) is our main contribution. It tries to predict the current image on the basis of an earlier image. To define which images are used, we denote DSN-_n_, where _n_ indicates how many images the DSN looks in the past. So if we have images _s_, _t_, _v_ and _w_ which occur sequentially (least to most present, _w_ is the current image), then DSN-2 takes image _t_.
The DSN-architecture is visualized below, with all four of its components. Here, the bold arrows are the in- and output of the DSN, while the dotted arrows describe actions which occur after an iteration.
![DSN visualization](/DSN/figures/DSN_visualization.png)
The code for the configuration can be found in `/DSN/config_DSN.py`, while the implementation of the DSN itself can be found here `DSN/DSN.py`.

## Overview of repository
Here, an overview of all directories is given. **Note that some scripts/code contain absolute/relative paths, so they might not be up to date with the repository. Change the paths manually when necessary.** Moreover, the datasets are not included.
<center>
| Directory    | Description                                                           |
|--------------|-----------------------------------------------------------------------|
| config       | `.ini`-files for configuring the DSN, and connecting to CoppeliaSim   |
| coppelia_sim | Scenes, `.lua`-code for robot/object and self-written API             |
| datasets     | Examples/overview of the used datasets, no complete datasets          |
| DEMO         | Two short videos of avoiding a moving obstacle                        |
| DSN          | DSN-class and code to configure the DSN                               |
| experiments  | Code + results for the two conducted experiments                      |
| models       | Baseline-class, all model architectures, weights and history-files    |
| thesis       | Thesis (`.pdf`) of project                                              |
</center>

## Usage
The models are trained on obstacle avoidance images with the classes: Backwards, Left, Right and Straight. The CNN its output is in the same order as mentioned here.

___

### Models
In order to use the baseline model, import `/models/baseline/baseline.py` and create a new instance of the Baseline class. Then:
``` 
model = Baseline(CNN_filename)  #create instance of baseline
prediction = model.determine_command(img)
```
Similarly, to use the DSN, import `/DSN/DSN.py` and create a new instance of the DSN class.
```
model = DSN(CNN_filename, DSN_variant, transformations, tau, resolution)  #create instance of DSN
prediction = model.determine_command(img)
```
Here, `CNN_filename` is the location where the weights are stored and `img` is a numpy array with shape `(1, resolution_y, resolution_x, 3)`. Then, the index denotes the command. Specifically for the DSN, `DSN_variant` contains _n_, `transformations` describes how every image transformation (zoom/shift) needs to be applied, `tau` is a parameter specific for the Command Vector and at lastly `resolution` denotes the resolution (both X and Y) of the input image.

The weights of the models are also included in the repository, and can be found under `/models/weights/`. It is also possible to train a fresh CNN with the files in `/models/baseline/modified/train_model/`. Here, for every dataset a script is written with which you can train the model from scratch.

___

### Configuring the DSN
But before you can use the DSN, it has to be configured to be able to construct realistic predictions. As earlier mentioned, the file which handles the configuration can be found in `/DSN/config_DSN.py`. Then, an `.ini`-file must be provided which defines the speed of the left and right motor for every command (Backwards, Left, Right, Straight). This file can be found here `/config/commands.ini`. **The order of defining the commands is important, do not change this.**

Now, with the use of the CoppeliaSim API, a small configuration run will be executed and the results will be written to a new `.ini`-file named `DSN.ini`. To run the program, use the following command:
```
python3 config_DSN.py [iter] [DSN-variant] [tau]
```
`iter` indicates how many runs will be executed for every command, the higher the number the more accurate it will be. 3 or 5 is already sufficient. The configuration makes use of AKAZE for feature point detection.

___

### CoppeliaSim API
An API has been written to connect the DSN/baseline to the CoppeliaSim simulator. This API can be found here: `coppelia_sim/API_coppeliasim.py`. It contains functions for the DSN configuration (`check_startup_sim` and `stop_simulation`), manually closing the connection with the simulator (`exit_API`) and provides and interface for CoppeliaSim. Here is a small example:
```
CS = CoppeliaSim(address, port)               #initialize and start the connection
CS.get_handles(motors, sensor)                #initialize the handles for motors/sensors
img = CS.get_image()                          #get image from robot

#do something with image

CS.set_velocity(v_left_motor, v_right_motor)  #set velocity for left and right motor
```
Moreover, the API requires three files, namely `remoteAPI.so`, `sim.py` and `simConst.py`. They have to be in the same directory (as is the case in the repository). Additionally, `/config/coppelia_sim.ini` contains the information about the address and port, along with the handles for various components of the robot and more. This file needs to be changed according to your own version of CoppeliaSim and robots.

At last, `.lua`-code has been provided with which moving obstacles (`/coppelia_sim/scripts/moving_obstacle.lua`) and an Obstacle Avoidance Dataset (`/coppelia_sim/scripts/pioneer.lua`) can be created.

___
