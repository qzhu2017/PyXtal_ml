from distutils.core import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="ml-materials",
    version="0.1dev",
    author="Qiang Zhu, Howard Yanxon, David Zagaceta",
    author_email="qiang.zhu@unlv.edu",
    description="Python code for Machine Learning of Materails' properties",
    long_description=long_description,
    url="https://github.com/qzhu2017/ML-Materials",
    #packages=['pyxtal', 'pyxtal.database'],
    #package_data={'pyxtal.database': ['*.csv', '*.json']},
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    requires=['pymatgen', 'numpy', 'scipy', 'sklearn', 'matplotlib'],
)
