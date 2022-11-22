# taichi-hackathon-akinasan
Akinasan team (秋名山车队)'s code base for the 0th Taichi Hackathon.

### Team name
Akinasan team (秋名山车队)

### Team member
[houkensjtu](https://github.com/houkensjtu), [Linus-Civil](https://github.com/Linus-Civil)

### Project name
Self-driven car powered by Taichi

### Project description
Our object is to control a RC car with neural networks trained for image classification written in the Taichi language. 
We plan to run our code on Nvidia Jetson Nano. 

As shown in the sketch below, we will use a camera module to capture images of the road, and the information will be processed
by a neural-network written in Taichi. The neural-network will output a decision to tell the RC car to turn left, turn right or
keep straight.

![project-overview](https://user-images.githubusercontent.com/2747993/203252767-d7b7613b-d85f-4fd7-a1ee-307466571c37.png)

[Taichi](https://docs.taichi-lang.org/) is an open source, high-performance parallel programming language embedded in Python. 
Thanks to its portability, Taichi can be run on various backends, including x86, CUDA, Arm and many other platforms. 

Nvidia Jetson Nano is a development board designed for running neural networks in parallel. 
It's equipped with an Arm Quad-core Cortex A57 CPU and a 128-core Maxwell GPU. 

### Hardware
We used the following hardwares in this project:
- Nvidia Jetson Nano 4GB
- Jetson Nano camera module
- Remote control car 
- L298N motor drive controller
- etc.

### Misc.
