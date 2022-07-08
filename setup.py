from setuptools import setup

setup(
    name='chroma',
    version='0.1',
    description='Chromatic PSFs',
    url='https://github.com/LSSTDESC/chroma',
    author='Josh Meyers',
    author_email='',
    license='MIT',
    packages=['chroma'],
    package_data={'chroma': ['data/filters/*.dat', 'data/SEDs/*.ascii', 'data/SEDs/*.spec', 'data/simard/*.fits']},
    python_requires='>=3.5'
)
