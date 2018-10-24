from setuptools import setup, find_packages
setup(
    name="demandtools",
    version="0.1dev",
    description="A package of demand estimators used in 'An analysis of consumer behaviour with FNNs",
    author="Serhat Ugurlu",
    license="MIT",
    packages=find_packages(exclude=["estimation_data.txt", "initial_codes.py"])
    ,
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'linearmodels'
    ],
)