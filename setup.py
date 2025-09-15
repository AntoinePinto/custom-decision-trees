from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="custom-decision-trees",
    version="2.0.0",
    description=(
        "A package for building customizable decision trees and random forests."
    ),
    author="Antoine Pinto",
    author_email="antoine.pinto1@outlook.fr",
    license="MIT",
    license_file="LICENSE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AntoinePinto/custom-decision-trees",
    project_urls={
        "Source Code": "https://github.com/AntoinePinto/custom-decision-trees",
    },
    keywords=[
        "machine learning",
        "decision trees",
        "random forest",
        "customization",
        "classification",
        "custom splitting criteria",
    ],
    packages=find_packages(),
    install_requires=[
        "joblib>=1.5.2",
        "matplotlib>=3.10.6",
        "numpy>=1.19.0",
        "tqdm>=4.50.0",
    ],
    python_requires=">=3.10",
)
