from glob import glob

from setuptools import setup


package_name = "control_pkg"

setup(
    name=package_name,
    version="0.0.1",
    py_modules=[
        "serial_bridge",
        "manual_cmd_vel",
    ],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="thislifewon",
    maintainer_email="user@example.com",
    description=(
        "Bridge steer angle / normalized throttle Float32 commands to "
        "the Dolsoi Arduino serial protocol."
    ),
    license="MIT",
    entry_points={
        "console_scripts": [
            "serial_bridge = serial_bridge:main",
            "manual_cmd_vel = manual_cmd_vel:main",
        ],
    },
)
