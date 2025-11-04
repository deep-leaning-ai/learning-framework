from setuptools import setup, find_packages

setup(
    name="learning-framework",
    version="0.1.0",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.10.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "keras": [
            "tensorflow>=2.10.0",
        ],
        "pytorch": [
            "torch>=1.13.0",
            "pytorch-lightning>=2.0.0",
        ],
        "huggingface": [
            "transformers>=4.25.0",
            "datasets>=2.8.0",
        ],
        "tracking": [
            "mlflow>=2.0.0",
            "wandb>=0.13.0",
        ],
    },
)
