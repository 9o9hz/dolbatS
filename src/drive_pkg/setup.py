from glob import glob

from setuptools import setup


package_name = "drive_pkg"

setup(
    name=package_name,
    version="0.1.0",
    py_modules=[
        "drive_pipeline",
        "lane_detect",
        "lane_processing",
        "path_plan",
        "path_visualizer",
        "pure_pursuit",
        "test_yolo_usb_cam",
        "yolo_lane_driver",
    ],
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml"]),
        (
            f"share/{package_name}/resource",
            ["resource/bev_params_0728.npz"],
        ),
        (
            f"share/{package_name}/config",
            glob("config/*.yaml"),
        ),
        (
            f"share/{package_name}/launch",
            glob("launch/*.launch.py"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="tak",
    maintainer_email="tak@example.com",
    description="Topic-based lane detection, path planning, and control.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "lane_detect = lane_detect:main",
            "path_plan = path_plan:main",
            "path_visualizer = path_visualizer:main",
            "pure_pursuit = pure_pursuit:main",
            "drive_pipeline = drive_pipeline:main",
            "yolo_lane_driver = drive_pipeline:main",
            "yolo_topic_test = test_yolo_usb_cam:main",
        ],
    },
)
