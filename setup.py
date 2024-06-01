from setuptools import setup, find_packages

setup(
    name='mimo_pack',
    description='Base code for the MIMO lab',
    author='Drew B. Headley',
    packages=find_packages(include=['mimo_pack', 'mimo_pack.*']),
)