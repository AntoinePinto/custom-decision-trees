from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="custom-decision-trees",
    version="2.0.2",
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
        "joblib>=1.4.0",
        "matplotlib>=3.9.0",
        "numpy>=1.26.0",
    ],
    python_requires=">=3.10",
)
