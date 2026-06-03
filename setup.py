from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

setup(
    name="GeoLens",
    version="1.0.0",
    description="Python library for satellite imagery processing and deep learning segmentation — built at ISRO",
    long_description=(here / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Janavi Nathwani",
    author_email="janavi.nathwani9@gmail.com",
    url="https://github.com/JanaviN7/GeoLens",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="satellite imagery, remote sensing, GeoTIFF, NISAR, SAR, deep learning, segmentation, ISRO",
    packages=find_packages(where="src"),
    python_requires=">=3.7, <4",
    install_requires=[
        "h5py",
        "numpy",
        "rasterio",
        "ipyleaflet",
        "shapely",
        "tqdm",
        "matplotlib",
        "scipy",
        "folium",
        "pyproj",
        "fiona",
        "geopandas",
        "tensorflow>=2.8",
        "keras",
        "opencv-python",
    ],
)
