from setuptools import setup, find_packages

setup(
    name="semantic_f1_score",
    version="1.0.1",
    description="Semantic F1 Score for Multi-label Classification",
    author="Georgios Chochlakis",
    author_email="chochlak@usc.edu",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scipy",
    ],
    extras_require={
        "dev": ["black", "pytest"],
        "test": ["pytest", "scikit-learn"],
    },
)
