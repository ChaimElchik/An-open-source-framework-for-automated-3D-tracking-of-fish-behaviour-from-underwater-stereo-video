# An open-source framework for automated 3D tracking of fish behaviour from underwater stereo video

This is a modular, open-source Python framework for automated 3D tracking of aquatic species from stereo video systems. It handles the full pipeline: from detecting and tracking fish in raw video footage to correcting for underwater refraction, matching stereo pairs, triangulating 3D coordinates, and generating ethological insights (velocity, acceleration, heatmaps).

## 🚀 Features

* **Robust 2D Tracking:** Uses YOLO + BoT-SORT with a custom Post-Track Re-ID module to heal broken tracks caused by occlusion.
* **Refractive Stereo Matching:** Implements a "Virtual Image Point" correction to account for light bending at the flat port interface, ensuring accurate 3D triangulation without complex ray-tracing.
* **Automated Metrics:** Calculates key biological indicators:
    * Instantaneous Velocity & Acceleration (with burst event detection).
    * Path Tortuosity & Cumulative Distance.
    * Volumetric Occupancy Heatmaps (3D space usage).
* **Visualization:** Generates annotated video overlays and 3D trajectory plots automatically.

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/ChaimElchik/An-open-source-framework-for-automated-3D-tracking-of-fish-behaviour-from-underwater-stereo-video
cd An-open-source-framework-for-automated-3D-tracking-of-fish-behaviour-from-underwater-stereo-video
```
### 2. Install Dependencies
It is recommended to use a Conda environment.
```bash
conda create -n deepdive python=3.9 -y
conda activate deepdive

# Install PyTorch (adjust cuda version for your GPU)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Install Framework Requirements
pip install ultralytics opencv-python pandas numpy scipy matplotlib seaborn tqdm pyyaml
```
### 3. File Structure
Ensure your project directory looks like this:
```bash
An-open-source-framework-for-automated-3D-tracking-of-fish-behaviour-from-underwater-stereo-video/
├── CustomeBoTSORT.yaml        # Tracker configuration
├── Run_PipeLine.py            # Main execution script
├── ProcessVideoPair.py        # YOLO + BoT-SORT + Re-ID logic
├── StereoMatching.py          # Refractive Epipolar Matching
├── ThreeDCordinate_Maker.py   # 3D Triangulation
├── InsightsGen.py             # Behavioral Analysis & Plotting
├── OutPutVideoGenerater.py    # Video Overlay Visualization
├── README.md
├── data/                      # (Optional) Store your videos/calib here
└──  ModelWeights/             # (Optional) Place your model weights here
```

### Cell 4: Usage Instructions
## ⚙️ Usage

You can run the entire pipeline using the `Run_PipeLine.py` script. You need a left video, a right video, and a stereo calibration file (`.mat`).

### Command Line Interface
```bash
python Run_PipeLine.py \
  --vid1 "path/to/left_camera.mp4" \
  --vid2 "path/to/right_camera.mp4" \
  --calib "path/to/stereoParams.mat" \
  --model "path/to/modeoweights.pt" \
  --output "./results"

Arguments
--vid1: Path to the video file from the Left Camera.

--vid2: Path to the video file from the Right Camera.

--calib: Path to the .mat file containing stereo calibration parameters (intrinsics, extrinsics, distortion).

--model: Path to the YOLO weights file (default: yolov8n.pt).

--output: Directory where all results (CSVs, plots, videos) will be saved.
```
### Cell 5: Pipeline & Outputs

## 📊 Pipeline Modules

The framework is divided into 5 sequential steps:

1.  **Detection & Tracking (`ProcessVideoPair.py`):** Runs YOLOv8 and BoT-SORT on both videos. Applies a custom Re-ID algorithm using kinematic logic and HSV color histograms to stitch fragmented tracks.
2.  **Stereo Matching (`StereoMatching.py`):** Matches fish IDs between Left and Right views using refractive-corrected epipolar geometry and the Hungarian algorithm.
3.  **3D Triangulation (`ThreeDCordinate_Maker.py`):** Converts matched 2D pixel coordinates into metric 3D (X, Y, Z) coordinates, applying Snell's Law to correct for air-glass-water refraction.
4.  **Behavioral Analysis (`InsightsGen.py`):** Smooths trajectories (Savitzky-Golay) and computes velocity, acceleration, and turn angles. Generates statistical reports and plots.
5.  **Visualization (`OutPutVideoGenerater.py`):** Renders the final tracking boxes and IDs back onto the original videos for verification.

## 📂 Outputs

The script creates a structured results folder:
```bash
results/ 
├── mots/ # Raw and Cleaned 2D tracking CSVs 
├── 3D/ # Final 3D Coordinate CSV (id, x, y, z, frame) 
├── analysis/ # Plots: Trajectories, Speed profiles, Heatmaps 
└── videos/ # Final annotated videos (Left & Right)
```

## 📝 Citation

If you use this code for your research, please cite:

> Chaim Chai Elchik, [Co-authors]. (2025). A Modular Framework for High-Precision 3D Tracking and Behavioral Analysis of Aquatic Species Using Refractive Stereo Vision*. Methods in Ecology and Evolution. [DOI link if available]

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.


