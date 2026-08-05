from glob import glob

from setuptools import setup


package_name = "detect_pkg"
package_share_files = ["package.xml"]

setup(
    name=package_name,
    version="0.0.1",
    py_modules=[
        "obstacle_detector_publisher",
        "traffic_light_detection",
        "ultrasonic_obstacle_event",
        "yolo_obstacle_bbox_turn",
        "yolo_obstacle_yolo_only",
        "yolo_obstacle_turn",
        "yolo_obstacle_fusion_end",
    ],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", package_share_files),
        (f"share/{package_name}/config", glob("config/*")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="thislifewon",
    maintainer_email="user@example.com",
    description="Obstacle and traffic light detectors subscribing to camera images.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "obstacle_detector_publisher = obstacle_detector_publisher:main",
            "traffic_light_detection = traffic_light_detection:main",
            "ultrasonic_obstacle_event = ultrasonic_obstacle_event:main",
            "yolo_obstacle_bbox_turn = yolo_obstacle_bbox_turn:main",
            "yolo_obstacle_yolo_only = yolo_obstacle_yolo_only:main",
            "yolo_obstacle_turn = yolo_obstacle_turn:main",
            "yolo_obstacle_fusion_end = yolo_obstacle_fusion_end:main",
        ],
    },
)
