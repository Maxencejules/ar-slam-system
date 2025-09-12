# AR SLAM System 
 
A real-time monocular visual SLAM (Simultaneous Localization and Mapping) system built 
from scratch in C++20, demonstrating core augmented reality tracking capabilities similar 
to those used in ARCore and ARKit. 
 
![C++](https://img.shields.io/badge/C%2B%2B-20-blue.svg) 
![OpenCV](https://img.shields.io/badge/OpenCV-4.6.0-green.svg) 
![OpenGL](https://img.shields.io/badge/OpenGL-3.3-red.svg) 
![License](https://img.shields.io/badge/license-MIT-purple.svg) 
 
## Overview 
 
This project implements fundamental AR tracking technology without using existing SLAM 
libraries, built specifically to demonstrate understanding of computer vision and AR 
systems for the Google AR Software Developer position. The system achieves real-time 
performance (46+ FPS) while tracking 1000+ features simultaneously with 3D visualization. 
 
## Key Features 
 
- **Real-time Performance**: 1000+ tracked features 
- **3D Point Cloud Visualization**: OpenGL-based rendering with depth-based coloring 
- **Robust Feature Tracking**: ORB feature detection with KLT optical flow
- **Modern C++ Design**: C++20 with smart pointers, templates, and RAII principles
- **Cross-platform Ready**: Designed for Linux with V4L2 camera interface structure  
- Real-time operation on standard hardware
- Robust tracking with automatic recovery
- Handles typical handheld camera motion
- Automatic re-detection when tracking degrades
- Memory-efficient design suitable for embedded systems

 
## System Architecture 
```
ar-slam-system/ 
├── Core Components (src/core/) 
│   ├── Frame Management 
│   ├── Feature Detection & Tracking 
│   └── Memory Management 
├── 3D Visualization (src/rendering/) 
│   ├── OpenGL 3.3 Core Profile 
│   └── Real-time point cloud rendering 
└── Testing Framework (tests/) 
    ├── Performance benchmarks 
    └── Unit tests 
```
 
## Quick Start 
 
### Prerequisites 
```bash
sudo apt-get update 
sudo apt-get install -y build-essential cmake libopencv-dev libglew-dev libglfw3-dev libglm-dev libeigen3-dev 
```
 
### Build Instructions 
```bash
# Clone repository 
git clone https://github.com/Maxencejules/ar-slam-system.git 
cd ar-slam-system 

# Create build directory 
mkdir build && cd build 

# Configure with CMake (Release mode for best performance) 
cmake .. -DCMAKE_BUILD_TYPE=Release 

# Build with all cores 
make -j$(nproc) 

# Run the main 3D tracking demo 
./src/camera_3d 
```
 
## Usage 
 
### Controls 
| Key | Action |
|-----|--------|
| Arrow Keys | Rotate/zoom 3D view |
| Space | Print performance statistics |
| ESC/Q | Exit application |
 
### Output Windows 
1. **2D Camera View**: Live camera feed with tracked features (green dots) 
2. **3D Point Cloud View**: Spatial visualization with depth-based coloring 
 
[GitHub Repository](https://github.com/Maxencejules/ar-slam-system)
 
## Technical Implementation 
 
### Feature Detection & Tracking 
- ORB Feature Detection: Up to 1000 features per frame 
- KLT Optical Flow: Pyramidal Lucas-Kanade with 3 levels 
- RANSAC: Outlier rejection with fundamental matrix estimation 
 
### Memory Pool Design 
Custom pool allocator with O(1) allocation/deallocation, maintaining 256MB constraint for 
embedded systems. 
 
### 3D Visualization 
OpenGL 3.3 Core Profile with custom shaders for point cloud rendering. Depth indicated 
by color gradient: 
- Near: Red 
- Mid: Green 
- Far: Blue 
 
## Dependencies 
 
| Library | Version | Purpose |
|---------|---------|---------|
| OpenCV  | 4.6.0   | Computer vision algorithms |
| OpenGL  | 3.3+    | 3D rendering |
| GLFW    | 3.3+    | Window management |
| GLEW    | 2.2+    | OpenGL extensions |
| GLM     | 0.9.9+  | Mathematics |
| Eigen3  | 3.4+    | Linear algebra |
 
## Project Structure 
```
ar-slam-system/ 
├── CMakeLists.txt 
├── README.md 
├── include/ 
│   ├── core/ 
│   │   ├── frame.h 
│   │   ├── feature_tracker.h 
│   │   └── memory_pool.h 
│   └── rendering/ 
│       └── gl_viewer.h 
├── src/ 
│   ├── camera_3d_test.cpp 
│   ├── core/ 
│   │   ├── frame.cpp 
│   │   └── feature_tracker.cpp 
│   └── rendering/ 
│       └── gl_viewer.cpp 
└── tests/ 
    ├── benchmark/ 
    │   └── benchmark_main.cpp 
    └── unit/ 
        └── test_features.cpp 
```
 
## Future Enhancements 
- Bundle adjustment for global optimization 
- Loop closure detection 
- Dense reconstruction 
- IMU sensor fusion 
- Mobile deployment (Android NDK) 
 
## License 
MIT License - see LICENSE file for details. 
 
## Contact 
Maxence Jules  
- Email: powe840@gmail.com  
- LinkedIn: linkedin.com/in/julesmax  
- GitHub: @Maxencejules  
 
Built with passion for augmented reality and computer vision.
