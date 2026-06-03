import os
import rasterio
import numpy as np
import folium
from pyproj import Transformer
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.crs import CRS
from rasterio.warp import reproject
from shapely.geometry import box, shape, Polygon
from affine import Affine
import fiona
import geopandas as gpd
import math
import matplotlib.pyplot as plt


class IsroGeoTiff:
    """
    Class for handling GeoTIFF files using rasterio.

    A high-level interface for reading, subsetting, tiling, masking,
    merging, and saving satellite raster data — built for ISRO data pipelines.

    Attributes:
        file_path (str): Path to the GeoTIFF file.
        crs (rasterio.crs.CRS): Coordinate reference system.
        transform (Affine): Transform matrix (image → map coordinates).
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        count (int): Number of bands.
        dtype (str): Pixel data type.
        data (np.ndarray): Image data as a NumPy array.
        title (str): Optional title.
        description (str): Optional description.
        parent (IsroGeoTiff): Parent object if this is a subset.
        children (list): Child objects if this has subsets.
    """

    def __init__(self, file_path=None, title='', description=None,
                 crs=None, transform=None, driver=None, width=None,
                 height=None, dtype=None, shape=None, count=None,
                 parent=None, children=None, data=None):
        self.file_path = file_path
        self.crs = crs
        self.transform = transform
        self.driver = driver
        self.width = width
        self.height = height
        self.dtype = dtype
        self.shape = shape
        self.count = count
        self.parent = parent
        self.children = children
        self.title = title
        self.description = description
        self.data = data

    # ------------------------------------------------------------------ #
    #  Metadata                                                            #
    # ------------------------------------------------------------------ #

    def set_metadata_from_tif(self):
        """Read and store metadata directly from the GeoTIFF file."""
        assert self.file_path and os.path.splitext(self.file_path)[-1].lower() in ['.tif', '.tiff'], \
            "Provided path is not a GeoTIFF file."
        with rasterio.open(self.file_path) as src:
            self.crs = src.crs
            self.transform = src.transform
            self.driver = src.driver
            self.width = src.width
            self.height = src.height
            self.shape = (src.height, src.width)
            self.dtype = src.meta['dtype']
            self.count = src.count

    def read_metadata(self):
        """Return a metadata dictionary from the GeoTIFF file."""
        try:
            with rasterio.open(self.file_path) as src:
                self.crs = src.crs
                self.transform = src.transform
                self.driver = src.driver
                self.width = src.width
                self.height = src.height
                self.shape = (src.height, src.width)
                return {
                    "Driver": self.driver,
                    "CRS": str(self.crs),
                    "Width": self.width,
                    "Height": self.height,
                    "Transform": str(self.transform),
                }
        except Exception as e:
            print(f"Error reading metadata: {e}")
            return None

    def set_data(self, data):
        """Set image data from a NumPy array and infer basic metadata."""
        if not isinstance(data, np.ndarray) or data.size == 0:
            print("Invalid or empty array.")
            return
        self.data = data
        self.height, self.width = data.shape[:2]
        self.count = 1
        self.dtype = str(data.dtype)
        self.crs = 'EPSG:4326'
        self.transform = rasterio.transform.from_origin(0, 0, 1, 1)

    def get_data(self):
        return self.data

    # ------------------------------------------------------------------ #
    #  Subsetting                                                          #
    # ------------------------------------------------------------------ #

    def raw_subset_from_tif(self, x, y, w, h, bands=1):
        """
        Extract a pixel-level rectangular subset.

        Args:
            x, y (int): Top-left corner in pixels.
            w, h (int): Width and height of the subset in pixels.
            bands (int): Band index to read.

        Returns:
            IsroGeoTiff: New object containing the subset.
        """
        try:
            assert os.path.exists(self.file_path)
            with rasterio.open(self.file_path) as src:
                window = rasterio.windows.Window(x, y, w, h)
                out_data = src.read(window=window, indexes=bands)
                if out_data.size == 0:
                    print("Empty subset.")
                    return None
                updated_transform = src.window_transform(window)
                return IsroGeoTiff(
                    title=f"{self.title}_subset_{x}_{y}_{w}_{h}",
                    description=self.description, crs=self.crs,
                    transform=updated_transform, width=w, height=h,
                    dtype=self.dtype, parent=self, count=bands, data=out_data
                )
        except Exception as e:
            print(f"Error in raw_subset_from_tif: {e}")

    def subset_from_shapefile(self, shapefile_gdf, bands=None):
        """
        Mask the raster to a shapefile geometry.

        Args:
            shapefile_gdf (GeoDataFrame): GeoDataFrame with a geometry column.
            bands (int or list): Bands to include.

        Returns:
            IsroGeoTiff: Masked subset.
        """
        geometry = shapefile_gdf['geometry'].iloc[0]
        with rasterio.open(self.file_path) as src:
            out_image, out_transform = mask(src, [geometry], indexes=bands, crop=True)
            is_multiband = len(out_image.shape) == 3
            if is_multiband:
                subset = IsroGeoTiff(
                    crs=src.crs, transform=out_transform, driver=src.driver,
                    width=out_image.shape[2], height=out_image.shape[1],
                    dtype=src.dtypes[0], count=bands
                )
            else:
                subset = IsroGeoTiff(
                    crs=src.crs, transform=out_transform, driver=src.driver,
                    width=out_image.shape[1], height=out_image.shape[0],
                    dtype=src.dtypes[0], count=1
                )
            subset.data = out_image
            return subset

    # ------------------------------------------------------------------ #
    #  Tiling                                                              #
    # ------------------------------------------------------------------ #

    def tile_image(self, file_path, tile_width, tile_height, output_directory):
        """
        Partition a large raster into fixed-size tiles for ML ingestion.

        Handles edge tiles automatically. Saves each tile as a GeoTIFF
        and returns the list of tile metadata dictionaries.

        Args:
            file_path (str): Path to the source raster.
            tile_width (int): Tile width in pixels.
            tile_height (int): Tile height in pixels.
            output_directory (str): Directory to save tile files.

        Returns:
            list[dict]: List of tile metadata dicts with keys:
                        data, transform, width, height.
        """
        try:
            tiles = []
            os.makedirs(output_directory, exist_ok=True)
            with rasterio.open(file_path) as src:
                width, height = src.width, src.height
                num_tiles_x = math.ceil(width / tile_width)
                num_tiles_y = math.ceil(height / tile_height)

                for i in range(num_tiles_x):
                    for j in range(num_tiles_y):
                        x_off = i * tile_width
                        y_off = j * tile_height
                        w = min(tile_width, width - x_off)
                        h = min(tile_height, height - y_off)
                        window = rasterio.windows.Window(x_off, y_off, w, h)
                        tile_data = src.read(window=window)

                        tile = {
                            "data": tile_data,
                            "transform": src.window_transform(window),
                            "width": w,
                            "height": h,
                        }
                        tiles.append(tile)

                        out_path = os.path.join(output_directory, f"Tile_{i}_{j}.tif")
                        with rasterio.open(
                            out_path, "w", driver="GTiff",
                            width=w, height=h, count=src.count,
                            dtype=tile_data.dtype, crs=src.crs,
                            transform=src.window_transform(window)
                        ) as dst:
                            dst.write(tile_data)

            print(f"Generated {len(tiles)} tiles → {output_directory}")
            return tiles
        except Exception as e:
            print(f"Error during tiling: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Merging                                                             #
    # ------------------------------------------------------------------ #

    def merge_geotiff_files(self, file_paths):
        """Merge multiple GeoTIFF files into a single data array."""
        datasets = [rasterio.open(fp) for fp in file_paths if os.path.exists(fp)]
        if datasets:
            merged_data, _ = merge(datasets, method='first')
            return merged_data
        return None

    # ------------------------------------------------------------------ #
    #  Masking                                                             #
    # ------------------------------------------------------------------ #

    def mask_raster_by_shapefile(self, raster_path, shapefile_path, output_directory):
        """Mask a raster by each polygon feature in a shapefile."""
        os.makedirs(output_directory, exist_ok=True)
        try:
            with rasterio.open(raster_path) as src:
                with fiona.open(shapefile_path, "r") as shp:
                    for i, feature in enumerate(shp):
                        geom = shape(feature["geometry"])
                        if not isinstance(geom, Polygon):
                            continue
                        out_image, out_transform = mask(src, [geom], crop=True)
                        out_meta = src.meta.copy()
                        out_meta.update({
                            "driver": "GTiff",
                            "height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform
                        })
                        out_path = os.path.join(output_directory, f"masked_{i}.tif")
                        with rasterio.open(out_path, "w", **out_meta) as dst:
                            dst.write(out_image)
                        print(f"Saved: {out_path}")
        except Exception as e:
            print(f"Error masking raster: {e}")

    # ------------------------------------------------------------------ #
    #  Saving                                                              #
    # ------------------------------------------------------------------ #

    def save(self, file_name, output_directory):
        """Save the image data as a GeoTIFF."""
        os.makedirs(output_directory, exist_ok=True)
        file_path = os.path.join(output_directory, file_name)
        try:
            if self.data is not None:
                with rasterio.open(
                    file_path, 'w', driver='GTiff',
                    width=self.width, height=self.height,
                    count=self.count, dtype=self.dtype,
                    crs=self.crs, transform=self.transform
                ) as dst:
                    dst.write(self.data, 1)
                print(f"Saved → {file_path}")
            else:
                print("No data to save.")
        except Exception as e:
            print(f"Save failed: {e}")

    def save_complex_tiff(self, file_name, output_directory):
        """Save two-band complex (real + imaginary) GeoTIFF."""
        os.makedirs(output_directory, exist_ok=True)
        file_path = os.path.join(output_directory, file_name)
        try:
            if self.data is not None and self.data.shape[0] == 2:
                with rasterio.open(
                    file_path, 'w', driver='GTiff',
                    width=self.width, height=self.height,
                    count=2, dtype=self.dtype,
                    crs=self.crs, transform=self.transform
                ) as dst:
                    dst.write(self.data, (1, 2))
                print(f"Saved complex GeoTIFF → {file_path}")
            else:
                print("Expected 2-band complex data.")
        except Exception as e:
            print(f"Save failed: {e}")

    # ------------------------------------------------------------------ #
    #  Visualisation                                                       #
    # ------------------------------------------------------------------ #

    def visualize_extent(self):
        """Render the image geographic extent on an interactive Folium map."""
        try:
            if self.crs and self.crs != CRS.from_epsg(4326):
                transformer = Transformer.from_crs(self.crs, "epsg:4326", always_xy=True)
                corners = [(0, 0), (0, self.width), (self.height, self.width), (self.height, 0)]
                reprojected = [transformer.transform(x, y) for x, y in corners]
                m = folium.Map(location=[0, 0], zoom_start=10)
                folium.Polygon(reprojected, color='blue').add_to(m)
                m.save('extent_map.html')
                print("Map saved → extent_map.html")
            else:
                print("CRS is already EPSG:4326 or not set.")
        except Exception as e:
            print(f"Visualisation error: {e}")

    def visualize_subset(self):
        """Display the subset data using Matplotlib."""
        if self.data is not None:
            plt.imshow(self.data, cmap='gray')
            plt.title(self.title or 'GeoTIFF Subset')
            plt.colorbar()
            plt.tight_layout()
            plt.show()
        else:
            print("No data to visualise.")
