from setuptools import setup, find_packages

with open('requirements.txt') as requirements_file:
    install_requirements = requirements_file.read().splitlines()

# modify entry_points to use command line 
# {COMMAND NAME}={module path}:{function in the module}
setup(
    name="histogram2vec",
    version="0.0.1",
    description="a package that converts histogram into vector",
    author="tadahaya",
    packages=find_packages(),
    install_requires=install_requirements,
    entry_points={
        "console_scripts": [
            "histogram2vec=histogram2vec.note_230913_01:main",
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3.9',
    ]
)