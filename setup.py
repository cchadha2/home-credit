import setuptools

setuptools.setup(
    name="home_credit",
    version="0.0.1",
    description="Training, model serving, and REST API code for Home-Credit model",
    packages=setuptools.find_packages(),
    install_requires=[
        "connexion[swagger-ui]",
        "lightgbm",
        "numpy",
        "pandas",
    ],
)
