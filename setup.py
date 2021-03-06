import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="clearskies",
    version="0.0.1",
    author="Luke Pratley",
    author_email="luke.pratley@gmail.com",
    description="Convex Denoiser",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Luke-Pratley/clearskies",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
          'numpy',
          'PyWavelets',
          'optimusprimal',
          'pytest',
          'astropy'
      ]
)

