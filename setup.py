import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="home_credit",
    version="0.0.1",
    description="Training, model serving, and REST API code for Home-Credit model",
    packages=setuptools.find_packages(),
    install_requires=[
        "lightgbm",
        "numpy",
        "pandas",
    ],
)
