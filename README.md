# CUIP ADS Evaluation GUI

This repository contains lightweight tools for running and visualizing Autoware ADS evaluation workflows.

## Repository Contents

- `autoware_bag_eval_gui.py`  
  GUI utility for launching/monitoring bag evaluation workflows.
- `record_autoware_eval_bag.sh`  
  Shell script for recording evaluation rosbags.
- `autoware_osm_heatmap_png.py`  
  Utility script to generate OSM-based heatmap PNG outputs.
- `test_route_03/metadata.yaml`  
  Example route metadata used by the evaluation tooling.

## Prerequisites

The exact runtime requirements depend on your environment and Autoware setup, but generally include:

- Linux environment with Bash
- Python 3.8+
- ROS 2 / Autoware workspace sourced correctly
- Access to map/route and bag data used for evaluation

## Quick Start

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd CUIP-ADS-evaluation-gui
```

### 2. (Optional) Create a Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 3. Run the tools

#### Launch the GUI

```bash
python3 autoware_bag_eval_gui.py
```

#### Record an evaluation bag

```bash
bash record_autoware_eval_bag.sh
```

#### Generate heatmap PNG output

```bash
python3 autoware_osm_heatmap_png.py
```

## Notes

- Ensure your ROS 2 and Autoware environment is sourced before running scripts.
- Update script parameters/paths as needed for your local map, route, and bag locations.
- If running in a container, verify display forwarding or headless configuration for GUI usage.

## Contributing

1. Create a feature branch.
2. Make focused changes with clear commit messages.
3. Open a pull request describing motivation and validation steps.

## License

Add a project license file (for example, `LICENSE`) and update this section accordingly.
