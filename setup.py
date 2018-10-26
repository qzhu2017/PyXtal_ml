from distutils.core import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyxtal_ml",
    version="0.0dev",
    author="Qiang Zhu, Howard Yanxon, David Zagaceta",
    author_email="qiang.zhu@unlv.edu",
    description="Python code for Machine Learning of crystal properties",
    long_description=long_description,
    url="https://github.com/qzhu2017/PyXtal-ml",
    packages=['pyxtal_ml', 'pyxtal_ml.datasets', 'pyxtal_ml.descriptors', 'pyxtal_ml.ml'],
    package_data={'pyxtal_ml.datasets': ['*.json'],
                  'pyxtal_ml.ml': ['*.yaml'],
                  'pyxtal_ml.descriptors': ['*.json', '*.yaml']},
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    requires=['pymatgen', 'numpy', 'scipy', 'sklearn', 'matplotlib'],
)
