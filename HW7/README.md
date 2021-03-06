# HW7: SLAM using GTSAM

In this assignment, I solved graph optimization problem using GTSAM for both 2D and 3D cases provided at https://lucacarlone.mit.edu/datasets/ . 

## Notes
I faced some issue for using Eigen::Matrix<double, 6, 6> member within a struct. This was caused by some memory alignment problem for a C++ struct. To solve this, we need to (1) either set C++ version up to 17, or (2) use Eigen::Matrix<double, 6, 6, Eigen::DontAlign> instead (to avoid auto alignment).

## Requirement
* C++ (preferred C++17 or more recent version)
* Eigen
* GTSAM

## Run
You will need four commands to run this project.  
`mkdir build`  
`cd build`  
`cmake ..`  
`make`  
`cd bin`  
`./hw7_##`  (## is the task number, ex: 1b, 2c)  

To plot the optimized trajectories, you'll need to go to the plot folder.  
`python3 plot_##.py` (## is the task number)  


## Results
![](https://i.imgur.com/mw55Jvh.png)
![](https://i.imgur.com/g31wqKU.png)
