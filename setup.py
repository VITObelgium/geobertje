from setuptools import find_packages, setup

setup(
    name="lithonlp",
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    description="Borehole Lithology Description NLP Python Package",
    author="VITO",
    # author_email='your@email.com',
    # url='https://github.com/yourusername/lithonlp',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        # Add your dependencies here
    ],
)
