from setuptools import setup


package_name = "camera_pkg"

setup(
    name=package_name,
    version="0.0.1",
    py_modules=["camera_publisher"],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="thislifewon",
    maintainer_email="user@example.com",
    description="Traffic-light and lane camera publishers.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "traffic_light_camera_publisher = camera_publisher:traffic_light_main",
            "lane_camera_publisher = camera_publisher:lane_main",
        ],
    },
)
