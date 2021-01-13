import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="resmem",
    version="1.0.0",
    author="Coën D. Needell",
    author_email="coen@needell.co",
    description="A package that wraps the ResMem pytorch model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Brain-Bridge-Lab/resmem",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'torch',
        'torchvision'
    ],
    python_requires='>=3.6',
)
