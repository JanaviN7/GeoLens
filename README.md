# 🛰️ GeoLens

> **Conference Presentation  - GeoLens ** — A Python library for satellite imagery processing and deep learning segmentation, developed at **ISRO** and presented at **ICAMADA 2024**.

[![Python](https://img.shields.io/badge/Python-3.7%2B-3670A0?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![IEEE](https://img.shields.io/badge/Published-IEEE%20ICAMADA%202024-00629B?style=flat-square&logo=ieee&logoColor=white)]()
[![ISRO](https://img.shields.io/badge/Built%20At-ISRO-FF6B35?style=flat-square)](https://www.isro.gov.in)
[![License](https://img.shields.io/badge/License-MIT-2ea44f?style=flat-square)](LICENSE)

---

## 📄 Conference Presentation

**"GeoLens: Enhanced Geo Dataset Orchestration Library for Pythonic Implementation"**

> Janavi Nathwani, Samvram Sahu, Jayasri PV, Usha Sundari HSV Rayali, Yerragudipadu Subbarayudu

- 📍 Presented at **ICAMADA 2024** (International Conference on Applied Mathematics and Advanced Data Analytics for Industry 5.0)
- 🏫 Department of CS Engineering – AI & ML, KG Reddy College of Engineering and Technology, Hyderabad
- 📦 Built in collaboration with **ISRO's Codelab** initiative

---

## 🌍 What Is GeoLens?

GeoLens is a Python library that simplifies geospatial raster data tasks and supports operations specific to **Remote Sensing satellite data** — particularly ISRO's **NISAR SAR** mission data.

It bridges the gap between complex satellite data and usable ML pipelines, making geospatial analysis accessible to researchers, professionals, and policymakers without deep domain expertise.

**Target users:** Environmental monitoring · Urban planning · Disaster management · Agricultural analysis

---

## ✨ Key Features

| Feature | Description |
|--------|-------------|
| 📂 **Metadata Reading** | Extract CRS, transform, band info from GeoTIFF files |
| ✂️ **Subset Creation** | Pixel-level and shapefile-masked subsets |
| 🔗 **Merging** | Combine multiple GeoTIFFs for time-series ARD applications |
| 🗺️ **Shapefile Masking** | Per-feature masking across vector boundaries |
| 🧩 **Tiling** | RasterTileDivider — partition large rasters into 256×256 ML-ready tiles |
| 💾 **Data Saving** | Export single-band and complex (real+imaginary) GeoTIFFs |
| 🛰️ **NISAR Support** | HDF5 → GeoTIFF conversion, extent visualisation, SAR calibration |
| 🤖 **Deep Learning** | 4 segmentation architectures for satellite scene understanding |

---

## 🧠 Deep Learning — Satellite Segmentation Models

Four architectures implemented and benchmarked for **8-class semantic segmentation** on 256×256 satellite imagery tiles:

| Model | Key Innovation | Architecture |
|-------|---------------|-------------|
| **ResUNet** | Residual blocks in encoder path | U-Net + residual skip connections |
| **U-Net** | Symmetric encoder-decoder | Classic segmentation baseline |
| **DeepLabV3+** | Atrous Spatial Pyramid Pooling (ASPP) | Depthwise separable conv + multi-scale context |
| **PSPNet** | Pyramid Pooling Module | Global + local context at scales 1, 2, 3, 6 |

All models: `Adam (lr=1e-4)` · `Categorical cross-entropy` · `ModelCheckpoint + EarlyStopping` callbacks

### ResUNet Architecture
```
Encoder: Conv → ResBlock(16) → Pool → ResBlock(32) → Pool → ResBlock(64) → Pool → ResBlock(128) → Pool → Conv(256)
Decoder: TransposeConv + SkipConcat → ×4
Output:  Softmax (8 classes)
```

### DeepLabV3+ — ASPP Module
```
ASPP: AvgPool branch + 3× Dilated Separable Conv branches (rates: 6, 12, 18)
→ Concat → Conv(256) → Upsample → Softmax (8 classes)
Output stride: 16
```

---

## 🛰️ NISAR SAR Data Processing

```python
from GeoLens import Nisar

nisar = Nisar()

# Visualise SAR acquisition extent on interactive map
nisar.visualize_extent("NISAR_product.h5")

# Convert NISAR HDF5 → GeoTIFF (RSLC + GSLC products)
nisar.write_radar_imagery("NISAR_product.h5", output_dir="./output")

# Generate calibrated backscatter image
calibrated = nisar.generate_calibrated_image(
    "NISAR_product.h5",
    parameter="sigma0",   # sigma0 / beta0 / gamma0
    polarization="HH"     # HH / HV / VV / VH / RH / RV
)

# Extract satellite pass metadata
metadata = nisar.get_identification_data("NISAR_product.h5")
# → track number, frame number, processing datetime, bounding polygon
```

---

## 🗺️ GeoTIFF Processing

```python
from GeoLens import IsroGeoTiff

tif = IsroGeoTiff(file_path="satellite_image.tif")
tif.set_metadata_from_tif()

# Pixel-level subset extraction
subset = tif.raw_subset_from_tif(x=100, y=100, w=512, h=512)
subset.visualize_subset()

# Mask by shapefile boundary
import geopandas as gpd
region = gpd.read_file("boundary.shp")
masked = tif.subset_from_shapefile(region, bands=[1, 2, 3])
masked.save("masked_output.tif", "./results")

# Tile large scene for ML pipeline
tiles = tif.tile_image(
    file_path="large_scene.tif",
    tile_width=256, tile_height=256,
    output_directory="./tiles"
)
print(f"Generated {len(tiles)} ML-ready tiles")
```

---

## 📦 Installation

```bash
git clone https://github.com/JanaviN7/GeoLens.git
cd GeoLens
pip install -e .
```

**Dependencies** (auto-installed via `setup.py`):
```
rasterio · numpy · tensorflow · keras · opencv-python
h5py · geopandas · shapely · folium · ipyleaflet
fiona · pyproj · scipy · tqdm · matplotlib
```

---

## 🏗️ Project Structure

```
GeoLens/
├── src/
│   ├── IsroGeoTiff.py       # GeoTIFF I/O, subsetting, tiling, masking
│   └── Nisar.py             # NISAR HDF5 SAR processing + calibration
├── models/
│   ├── resunet.py           # ResUNet — residual encoder-decoder
│   ├── unet.py              # U-Net — classic segmentation baseline
│   ├── deeplabv3.py         # DeepLabV3+ — ASPP + depthwise separable conv
│   └── pspnet.py            # PSPNet — pyramid pooling (2 variants)
├── setup.py
└── README.md
```

**Supported formats:** GeoTIFF · NISAR HDF5 (RSLC/GSLC) · Shapefiles · Any rasterio-compatible CRS

---

## 📊 Impact

- **30% improvement** in satellite data analysis efficiency vs. manual workflows
- Supports all **microwave sensor satellites** — versatile across missions
- Addresses ISRO's open data policy by lowering the barrier to satellite data analysis
- AutoML pipeline under development for automated model selection on SAR datasets

---

## 📎 Citation

```
J. Nathwani et al., "GeoLens: Enhanced Geo Dataset Orchestration Library 
for Pythonic Implementation," ICAMADA 2024.
```

> 📁 Trained model weights + sample NISAR data available on request
> 📧 [janavi.nathwani9@gmail.com](mailto:janavi.nathwani9@gmail.com)

---

## 🤝 Author

**Janavi Nathwani** — AI Engineer, Hyderabad

Built Ventsa (Voice AI SaaS) · IEEE Published Researcher · ISRO ML Intern · LLM Integration

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat-square&logo=linkedin)](https://linkedin.com/in/jahnavi-nathwani)
[![Email](https://img.shields.io/badge/Email-Hire%20Me-EA4335?style=flat-square&logo=gmail)](mailto:janavi.nathwani9@gmail.com)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
