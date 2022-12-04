# taichi-hackathon-akinasan
Akinasan team (秋名山车队)'s code base for the 0th Taichi Hackathon.

### Team name
Akinasan team (秋名山车队)

### Team member
[houkensjtu](https://github.com/houkensjtu), [Linus-Civil](https://github.com/Linus-Civil)

### Project name
Self-driving car powered by Taichi

### Project description
Our goal is to build a self-driving car based on Nvidia Jetson Nano. The project proposal is as follows:
1. The circuit of an ordinary RC toy car will be modified so that Jetson Nano can control the movement of the car through GPIO port. Of course, we need to use motor drive controller here, because the upper limit of the output current of Jetson Nano is not enough to drive the car motor directly.
2. The convolution neural network (CNN) will be implemented based on Taichi programming language.
3. The road data will be collected, then classified and labeled, and finally used in the training of CNN models.
4. The pre-trained model will be imported into Jetson Nano and the action prediction will be made for the images captured during driving.

 The running procedure of the self-driving car is shown in the sketch below. All the hardware will be fixed to the modified car. The image is captured by the camera module and passed into the CNN model running on Jetson Nano. The model predicts the corresponding behavior of the image. Finally, according to the predicted results, Jetson Nano sends out logic signals through the GPIO port to control the movement of the car.

![image](https://user-images.githubusercontent.com/46706788/205470986-79449846-175e-46f0-ae98-9b6f438aa025.png)

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
