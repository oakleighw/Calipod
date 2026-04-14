
<div align="center">

# Calipod

*Multicamera Planning, Calibration & Triangulation Gui for Insect Tracking, Derived from Caliscope - Open Source Motion Capture*

[![GitHub last commit](https://img.shields.io/github/last-commit/oakleighw/calipod.svg)](https://github.com/oakleighw/calipod/commits)
[![GitHub stars](https://img.shields.io/github/stars/oakleighw/calipod.svg?style=social&label=Star)](https://github.com/oakleighw/calipod/stargazers)
[![License: BSD 2-Clause](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
![pytest](https://github.com/oakleighw/calipod/actions/workflows/pytest.yml/badge.svg)
</div>

## Attribution

**Calipod** is derived from [Caliscope](https://github.com/mprib/caliscope), an open-source multicamera calibration and motion capture tool created by Mac Prible. Calipod extends the foundational calibration methodology with additional features for arena simulation, camera planning, detection filtering, YOLO detection loading, background subtraction support, and other optimizations for insect behaviour tracking. The underlying bundle adjustment and calibration algorithms are based on Caliscope's work, which is documented in the [Caliscope JOSS publication](https://joss.theoj.org/papers/10.21105/joss.07155).

---

## What's New in Calipod

Calipod extends Caliscope's core calibration with specialized features for multi-animal tracking workflows:

- **Arena Simulation & Planning** - Interactive 3D workspace design with triangulatable camera frustum overlap analysis
- **Advanced Post-Processing** - Trajectory smoothing, gap-filling, and Kalman filtering with customizable parameters
- **YOLO Integration** - Support for custom ground truth and detection predictions from YOLO detection systems
- **Background Subtraction** - Specialized processing for motion isolation workflows
- **Full Tracking Pipeline Guidance** - Step-by-step GUI workflow from calibration through post-processing

---

## About

`Calipod` is a GUI-based multicamera planning, calibration and motion tracking package, designed for insect behaviour analysis. It provides tools for camera planning optimization, intrinsic and extrinsic camera calibration, and 3D landmark triangulation.

Calipod automates complex calibration functions while providing visual feedback regarding parameter estimates at each stage of processing. Sample implementations of a Tracker class using Google's Mediapipe demonstrate how to integrate full calibration results with landmark tracking tools to achieve 3D pose estimation. Mediapipe capabilities are retained in the tracking pipeline, allowing users to test their camera rig and calibration by tracking their hand before deploying to insect tracking. While Mediapipe has limitations, it demonstrates a data processing pipeline easily extensible with more powerful tracking tools.

For a detailed description of the calibration process and workflow, see the [original Caliscope documentation](https://mprib.github.io/caliscope/). For a quick video walkthrough, check out Caliscope's [sample project demonstration](https://www.youtube.com/watch?v=voE3IKYtuIQ).

---

## Key Features

### Core Calibration Features

#### Calibration Board Creation
- Easy creation of `png` / `pdf` files for ChArUco calibration boards 
- Board definition can be changed across intrinsic and extrinsic calibration allowing greater flexibility

#### Intrinsic Camera Calibration
- Automated calculation of camera intrinsic properties from input video
  - Optical Center
  - Focal Length
  - Lens Distortion
- Visualization of distortion model to ensure reasonableness

#### Extrinsic Camera Calibration
- Automated bundle adjustment to estimate 6 DoF relative position of cameras
- Visualizer to inspect the estimates from the bundle adjustment
- Setting of the World Origin within the visualizer to simplify data processing

#### 3D Tracking
- Tracker API for integrating alternate tracking methods
  - 3 sample implementations with Google Mediapipe (Hands/Pose/Holistic)
- Automated application of landmark tracking to synchronized videos
- Triangulation of 3D landmark position based on calibrated cameras

#### Trajectory Output
- Output to `.trc` file format for use in biomechanical modelling
- Output to tidy `.csv` format with well-labelled headers for straightforward integration with other workflows
- Compatible with companion project [Rigmarole](https://github.com/mprib/rigmarole) (a Caliscope side-project for creating animated rigs in Blender)

### Extended Features for Insect Tracking

#### Arena Simulation & Planning
- Interactive 3D workspace visualization for camera placement optimization
- Triangulatable camera frustum overlap analysis to identify recording zones with sufficient multi-camera coverage
- Workspace validation and planning before hardware deployment

#### Advanced Post-Processing & Filtering
- RTS (Rauch-Tung-Striebel) Kalman filtering for 3D trajectory smoothing
- Intelligent gap-filling to handle occlusion or tracking failures
- Configurable Kalman filter parameters (process noise and measurement uncertainty)
- Distance-based spatial filtering with gate distance and threshold customization
- Performance metrics and visualization for filter validation

#### Background Subtraction Support
- Integrated background subtraction processing for scenarios requiring motion isolation
- Specialized triangulation workers for background-subtracted tracking workflows

#### Enhanced Charuco Board Creation
- Flexible calibration board configuration with custom PDF size selection
- Board parameter management for reproducible calibration across sessions

#### Metadata Persistence
- Automatic saving of filter parameters and calibration metrics alongside processed data
- Enables reproducibility and detailed tracking of processing history

---

# Quick Start

Calipod is compatible with Python 3.10 and 3.11. Please note that given the size of some core dependencies (OpenCV, Mediapipe, and PySide6 are among them) installation and initial launch can take a while.

## Installation Steps

### 1. Install prerequisite packages (Linux/Ubuntu only)
```
sudo apt-get update
sudo apt-get install --fix-missing libgl1-mesa-dev
```

### 2. Navigate to your project directory
**Windows:**
```
cd path\to\your\project
```

**Linux/macOS:**
```
cd path/to/your/project
```

### 3. Create a virtual environment
```
uv venv --python 3.11
```

### 4. Activate the virtual environment
**⚠️ Do this each time you use Calipod:**

**Windows:**
```
.\venv\Scripts\activate
```

**Linux/macOS:**
```
source .venv/bin/activate
```

### 5. Install Calipod
With your virtual environment activated and in the calipod directory, install Calipod using uv:
```
uv pip install -e .
```
Installation may take a moment as some dependencies are large, but uv's performance makes this process significantly faster than traditional tools.

## Project Directory Structure

For help organizing your project, see the [directory layout guide](https://mprib.github.io/caliscope/sample_project/) from the original Caliscope documentation (does not include annotation folder etc though, which is required for custom detection visualisation):

```
ProjectDirectory/
├── config.toml    # Only contains default charuco board definition
├── calibration/
│   ├── intrinsic/
│   └── extrinsic/
├── annotations/
│   ├── ground_truth/         # Ground truth YOLO annotations
│   │   ├── port_1/
│   │   │   ├── labels/train/ [txt files in YOLO format]
│   │   ├── port_2/
│   │   │   ├── labels/train/ [txt files in YOLO format]
│   ├── predictions/          # Optional YOLO predictions from models
│   │   ├── port_1/
│   │   │   ├── labels/ [txt files in YOLO format]
│   │   ├── port_2/
│   │   │   ├── labels/ [txt files in YOLO format]
└── recordings/               # Empty by default prior to user populating data
    ├── recording_1
    │   ├── port_1.mp4...
```

## Current Tracker Implementation

Calipod currently uses Google's Mediapipe for tracking, which provides a relatively easy and efficient method for pose estimation. However, Mediapipe has limitations for many applications. Following Caliscope's modular Tracker API, Calipod is designed to be extensible with more powerful tools such as [MMPose](https://github.com/open-mmlab/mmpose) and [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) as they are integrated in future updates.

## Reporting Issues and Requesting Features

To report a bug or request a feature, please [open an issue](https://github.com/oakleighw/calipod/issues).

## General Questions and Conversation

Post any questions in the [Discussions](https://github.com/oakleighw/calipod/discussions) section of the repo. 


# Acknowledgments

**Calipod** extends and builds upon the work of [Caliscope](https://github.com/mprib/caliscope), created by Mac Prible as an alternative calibration tool within the broader ecosystem of motion capture software. 

The original Caliscope project was inspired by [FreeMoCap](https://github.com/freemocap/freemocap) (FMC), which is spearheaded by [Jon Matthis, PhD](https://jonmatthis.com/) of the HuMoN Research Lab. The FMC calibration and triangulation system is built upon [Anipose](https://github.com/lambdaloop/anipose), created by Lili Karushchek, PhD. Caliscope was originally envisioned as an alternative calibration tool to Anipose that would allow more granular estimation of intrinsics as well as visual feedback during the calibration process. Several lines of the original Anipose triangulation code are used in this code base, though otherwise it was written from the ground up. 

# License

Calipod is licensed under the permissive [BSD 2-Clause license](https://opensource.org/license/bsd-2-clause/), as is the original [Caliscope](https://github.com/mprib/caliscope) project. This allows you to freely use, modify, and distribute Calipod, provided you retain the copyright and license notices.

Copyright holders:
- **2024**: Donald "Mac" Prible (Caliscope)
- **2025-2026**: Oakleigh Weekes (Calipod)

The triangulation function was adapted from the [Anipose](https://github.com/lambdaloop/anipose) code base which is also licensed under the BSD-2 Clause. A primary dependency of this project is PySide6 which provides the GUI front end. PySide6 is licensed under the [LGPLv3](https://www.gnu.org/licenses/lgpl-3.0.html). Calipod does not modify the underlying source code of PySide6 which is available via [PyPI](https://pypi.org/project/PySide6/).
