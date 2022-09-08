"""POSGGym package install file."""
from setuptools import setup, find_packages

# Environment and test specific dependencies.
extras = {
    "test": ["pytest>=6.2"],
    "render": ["matplotlib>=3.5"],
    "highwayenv": ["highway-env==1.6"]
}

extras['all'] = [item for group in extras.values() for item in group]


setup(
    name='posggym',
    version='0.0.1',
    url="https://github.com/RDLLab/posggym/",
    description=(
        "Partially Observable Stochastic Game environments with models."
    ),
    author="RDLLab",
    author_email="Jonathon.Schwartz@anu.edu.au",
    license="MIT",
    packages=[
        package for package in find_packages()
        if package.startswith('posggym')
    ],
    install_requires=[
        'gym>=0.21',
        'numpy>=1.20',
    ],
    extras_require=extras,
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    zip_safe=False
)
