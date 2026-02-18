from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="audio-spoofing-detection",
    version="2.0.0",
    author="Audio Spoofing Detection Team",
    description="Concatenative audio spoofing detection using frame-level boundary detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://huggingface.co/spaces/ujalaarshad17/AudioSpoofing",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "audio-spoofing-train=scripts.run_training:main",
            "audio-spoofing-infer=scripts.run_inference:main",
        ],
    },
)
