from setuptools import setup, find_packages

setup(
    name="quantsystem",
    version="0.1.0",
    description="量化交易系统 - Quantitative Trading System",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "yfinance>=0.2.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0"],
    },
)
