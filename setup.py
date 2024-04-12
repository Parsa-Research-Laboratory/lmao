from setuptools import setup

if __name__ == "__main__":
    setup(
        name="lmaao",
        version="0.0.1",
        install_requires=[
            "h5py",
            "ipykernel",
            "matplotlib",
            "numpy",
            "omegaconf",
            "scikit-optimize",
            "tqdm",
        ],
        author="Mavi, Derek, Shay",
        author_email="ssnyde9@gmu.edu",
        description = "Lava Multi-Agent Asynchronous Optimization in Lava",
        maintainer="Shay Snyder, Derek Gobin, Maria Victoria Clerico",
        maintainer_email="ssnyde9@gmu.edu",
        packages=["lmaao"]
    )