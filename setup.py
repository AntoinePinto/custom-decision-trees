from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="custom-tree-classifier",
    version="1.0.2",
    description=(
        "A package for building decision trees and random forests with "
        "custom splitting criteria."
    ),
    author="Antoine PINTO",
    author_email="antoine.pinto1@outlook.fr",
    license="MIT",
    license_file="LICENSE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AntoinePinto/custom-tree-classifier",
    project_urls={
        "Source Code": "https://github.com/AntoinePinto/custom-tree-classifier",
    },
    keywords=[
        "decision tree",
        "random forest",
        "machine learning",
        "classification",
        "custom splitting criteria"
    ],
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "tqdm>=4.50.0"
    ],
    python_requires=">=3.10"
)
