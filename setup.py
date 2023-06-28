from setuptools import find_packages, setup
from taxi_travel_regressor import __version__

setup(
    name='taxi_travel_regressor',
    packages=find_packages(exclude=['tests', 'tests.*']),
    setup_requires=['wheel'],
    version=__version__,
    description='Demo repository implementing NYC Taxi travel time regression model',
    authors='Julie Nguyen'
)
