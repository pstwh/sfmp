from setuptools import setup

setup(
    name="sfmp",
    version="1.0.1",
    packages=["sfmp"],
    include_package_data=True,
    url="https://github.com/psthw/sfmp",
    keywords="separate, images, automatically, cluster, feature, deeplearning",
    package_data={"sfmp": ["artifacts/*.onnx"]},
    python_requires=">=3.5, <4",
    install_requires=[
        "pillow==10.4.0",
        "scikit-learn==1.5.1",
        "tqdm==4.66.4",
    ],
    extras_require={
        'cpu': [
            'onnxruntime==1.18.1',
        ],
        "gpu": [
            "onnxruntime-gpu==1.18.1",
        ],
    },
    entry_points={
        "console_scripts": ["sfmp=sfmp.cli:main"],
    },
    description="Separate images automatically",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
