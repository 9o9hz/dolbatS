from glob import glob

from setuptools import setup


package_name = "detect_pkg"

setup(
    name=package_name,
    version="0.0.1",
    py_modules=["obstacle_detector_publisher"],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml", "dolsoi-model-v2.pt"]),
        (f"share/{package_name}/config", glob("config/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="thislifewon",
    maintainer_email="user@example.com",
    description="Obstacle detector subscribing to camera images.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "obstacle_detector_publisher = obstacle_detector_publisher:main",
        ],
    },
)
