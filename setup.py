from setuptools import setup

setup(
    name='chroma',
    version='0.1',
    description='Chromatic PSFs',
    url='https://github.com/LSSTDESC/chroma',
    author='Josh Meyers',
    author_email='jmeyers314@gmail.com',
    maintainer='Sidney Mau',
    maintainer_email='sidneymau@gmail.com',
    license='MIT',
    packages=['chroma'],
    package_data={'chroma': ['data/filters/*.dat', 'data/SEDs/*.ascii', 'data/SEDs/*.spec', 'data/simard/*.fits']},
    python_requires='>=3.5'
)
