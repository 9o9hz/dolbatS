from glob import glob

from setuptools import setup


package_name = "drive_pkg"

setup(
    name=package_name,
    version="0.1.0",
    py_modules=[
        "drive_main",
        "lane_detect",
        "lane_processing",
        "pure_pursuit",
        "visualizer",
        "yolotl",
    ],
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml"]),
        (
            f"share/{package_name}/resource",
            [
                "resource/bev(0729).npz",
                "resource/best.pt",
                "resource/bev_params_0731.npz",
            ],
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
            "drive_main = drive_main:main",
            "pure_pursuit = pure_pursuit:main",
            "yolotl = yolotl:main",
        ],
    },
)
