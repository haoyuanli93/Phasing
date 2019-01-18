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
    install_requires=['numpy>=1.10',
                      'matplotlib',
                      'h5py',
                      'numba',
                      'cudatoolkit',
                      'pyculib',
                      'scipy'],
    platforms=["Linux"],
    url='https://github.com/haoyuanli93/DiffusionMap'
)
