from glob import glob

from setuptools import setup

package = "tb4_emotion_control"
setup(
    name=package,
    version="0.1.0",
    packages=[package],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package]),
        ("share/" + package, ["package.xml"]),
        ("share/" + package + "/launch", glob("launch/*.launch.py")),
        ("share/" + package + "/config", glob("config/*.yaml")),
        ("share/" + package + "/worlds", glob("worlds/*.sdf")),
        ("share/" + package + "/models", glob("models/*.onnx")),
    ],
    install_requires=["setuptools", "numpy"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="HRI Lab",
    maintainer_email="hri@example.com",
    description="Facial expression recognition (HSEmotion/DeepFace/FER) driving TurtleBot 4 motion via /cmd_vel",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "emotion_detector_node = tb4_emotion_control.emotion_detector_node:main",
            "emotion_motion_node = tb4_emotion_control.emotion_motion_node:main",
            "emotion_sequence = tb4_emotion_control.emotion_sequence:main",
            "trajectory_recorder = tb4_emotion_control.trajectory_recorder:main",
            "emotion_calibrate = tb4_emotion_control.emotion_calibrate:main",
            "demo_recorder = tb4_emotion_control.demo_recorder:main",
        ]
    },
)
