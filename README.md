# yolo_v4_demo

This is a yolo_v4 detection demo using c++ and libdarknet.so with camera

**Authors:** cvyang

## 1. Prerequisites

### 1.1 requirements
  * Ubuntu 16.04/Ubuntu 18.04
  * opencv 
  * CUDA is not neccessary !!!
  * C++11/14
  * GCC
  * cmake
  * OpenCV3 or OpenCV4

### 1.2 Compile

git clone https://github.com/AlexeyAB/darknet.git

copy darknet/cfg coco.names, cfg file and weights file to yolo_v4_demo

git clone https://github.com/CVYang/yolo_v4_demo.git

copy libdarknet.so to /usr/lib

cmake .

make

## 2. Run the code

./yolo_v4 

## 3. notice
 you can edit the main.cpp for camera,image or video
 if your opencv version is not opencv 4.x, you will edit the CMakeLists.txt and rebuild the project





