from setuptools import setup


package_name = "control_pkg"

setup(
    name=package_name,
    version="0.0.1",
    py_modules=["serial_bridge"],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="thislifewon",
    maintainer_email="user@example.com",
    description="Bridge cmd_vel to the Dolsoi Arduino serial protocol.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "serial_bridge = serial_bridge:main",
        ],
    },
)
