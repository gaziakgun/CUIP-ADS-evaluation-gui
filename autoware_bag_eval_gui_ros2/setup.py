from glob import glob
import os

from setuptools import find_packages, setup


package_name = "autoware_bag_eval_gui_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [os.path.join("resource", package_name)]),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob(os.path.join("launch", "*.launch.py"))),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="gazi",
    maintainer_email="gazi@example.com",
    description="Autoware ROS 2 bag evaluation GUI with OSM heatmaps, plots, and PDF export.",
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "autoware_bag_eval_gui = autoware_bag_eval_gui_ros2.gui:main",
        ],
    },
)
