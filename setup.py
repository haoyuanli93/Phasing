from setuptools import setup

setup(
    name='phase',
    version='dev',
    description=(
        'This package contains different phase retrieval algorithms. '),
    long_description=("In this package, I try to implement all phase "
                      "retrieval algorithms for both cpu and gpu computations."),
    author='Haoyuan Li',
    author_email='hyli16@stanford.edu',
    maintainer='Haoyuan Li',
    maintainer_email='hyli16@stanford.edu',
    license='BSD License',
    packages=["phase", ],
    install_requires=['numpy',
                      'matplotlib',
                      'h5py'],
    platforms=["Linux"],
    url='https://github.com/haoyuanli93/DiffusionMap'
)
