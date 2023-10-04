from setuptools import setup, find_packages
import harissa

setup(

    name='harissa',

    version=harissa.__version__,

    packages=find_packages(),

    author='Ulysse Herbach',

    author_email='ulysse.herbach@inria.fr',

    description=('Tools for mechanistic gene network inference '
        'from single-cell data'),

    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',

    url='https://github.com/ulysseherbach/harissa',

    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],

    platforms='any',

    license='BSD-3-Clause',
    license_file='LICENSE.txt',

    keywords=('stochastic gene expression, gene regulatory networks, '
        'single-cell transcriptomics'),

    python_requires='>=3.8',

    install_requires=[
        'numpy>=1.20',
        'scipy>=1.7',
        'numba>=0.55',
        'matplotlib>=3.4',
        'networkx>=2.6',
    ],

)
