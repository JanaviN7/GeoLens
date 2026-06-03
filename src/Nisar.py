import os
import h5py as hp
import numpy as np
import rasterio as rio
from matplotlib import pyplot as plt
from glob import glob
from ipyleaflet import Map, WKTLayer
import shapely.wkt as swkt
from tqdm.auto import tqdm
from scipy.interpolate import RegularGridInterpolator


class Nisar:
    """
    Processor for ISRO/NASA NISAR SAR (Synthetic Aperture Radar) HDF5 products.

    Supports RSLC and GSLC product levels for both LSAR and SSAR instruments.
    Handles multi-polarization data (HH, HV, VV, VH, RH, RV) and generates
    calibrated backscatter imagery (sigma0, beta0, gamma0).

    Usage:
        nisar = Nisar()
        nisar.write_radar_imagery("NISAR_product.h5", output_dir="./output")
        calibrated = nisar.generate_calibrated_image("NISAR_product.h5", parameter="sigma0")
    """

    def __init__(self, folders=None, sars=None, products=None,
                 swaths=None, frequency=None, polarization=None,
                 output_dir='./NISAR_TIF_OUT'):
        self.folders = folders or ["science"]
        self.sars = sars or ["LSAR", "SSAR"]
        self.products = products or ["RSLC", "GSLC"]
        self.swaths = swaths or ["swaths", "grids"]
        self.frequency = frequency or ["frequencyA", "frequencyB"]
        self.polarization = polarization or ["HH", "HV", "VV", "VH", "RH", "RV"]
        self.output_dir = output_dir
        self.cal_dict = {}

    def visualize_extent(self, rslc_file):
        """Render SAR acquisition bounding polygon on an interactive map."""
        rslc_h5 = hp.File(rslc_file, 'r')
        identification = rslc_h5['science/LSAR/identification']
        bounding_polygon = swkt.loads(identification['boundingPolygon'][()].decode())
        scene_center = bounding_polygon.centroid.coords[0]
        m = Map(center=scene_center[::-1], zoom=8)
        m.add_layer(WKTLayer(wkt_string=bounding_polygon.wkt))
        display(m)

    def get_tiff_combinations(self, rslc_file):
        """
        List all available SAR data paths and output TIFF filenames in the HDF5.

        Returns:
            list[dict]: Each dict has 'datapath' (HDF5 path) and 'filepath' (output name).
        """
        combination = []
        rslc_h5 = hp.File(rslc_file, 'r')
        for folder in self.folders:
            if folder not in rslc_h5: continue
            for sar in self.sars:
                if sar not in rslc_h5[folder]: continue
                sar_data = rslc_h5[folder][sar]
                id_data = sar_data['identification']
                for product in self.products:
                    if product not in sar_data: continue
                    for swath in self.swaths:
                        if swath not in sar_data[product]: continue
                        for frq in self.frequency:
                            if frq not in sar_data[product][swath]: continue
                            for pol in self.polarization:
                                if pol not in sar_data[product][swath][frq]: continue
                                track = id_data["trackNumber"][()]
                                frame = id_data["frameNumber"][()]
                                date = id_data["processingDateTime"][()].decode().split("T")[0].replace("-", "")
                                fname = f"{track}_{frame}_{date}_{sar}_{product}_{frq[-1]}_{pol}.tif"
                                combination.append({
                                    'datapath': sar_data[product][swath][frq][pol].name,
                                    'filepath': fname
                                })
        return combination

    def write_radar_imagery(self, rslc_file, output_dir='./NISAR_TIF_OUT'):
        """
        Convert NISAR HDF5 products to GeoTIFF format.

        Handles both RSLC (complex: real + imaginary bands) and
        GSLC (single float band) product levels.

        Args:
            rslc_file (str): Path to NISAR HDF5 file.
            output_dir (str): Output directory for GeoTIFF files.

        Returns:
            dict: Nested dict mapping SAR → product → frequency → output path.
        """
        os.makedirs(output_dir, exist_ok=True)
        combination = {}
        rslc_h5 = hp.File(rslc_file, 'r')

        for folder in self.folders:
            if folder not in rslc_h5: continue
            for sar in self.sars:
                if sar not in rslc_h5[folder]: continue
                sar_data = rslc_h5[folder][sar]
                combination[sar] = {}
                id_data = sar_data['identification']
                for product in self.products:
                    if product not in sar_data: continue
                    combination[sar][product] = {}
                    for swath in self.swaths:
                        if swath not in sar_data[product]: continue
                        for frq in self.frequency:
                            if frq not in sar_data[product][swath]: continue
                            freq_data = sar_data[product][swath][frq]
                            combination[sar][product][frq] = {}
                            for pol in self.polarization:
                                if pol not in freq_data: continue
                                track = id_data["trackNumber"][()]
                                frame = id_data["frameNumber"][()]
                                date = id_data["processingDateTime"][()].decode().split("T")[0].replace("-", "")
                                fname = f"{track}_{frame}_{date}_{sar}_{product}_{frq[-1]}_{pol}.tif"
                                out_path = os.path.join(output_dir, fname)
                                combination[sar][product][frq][pol] = out_path
                                img = freq_data[pol][:, :]
                                if product == 'RSLC':
                                    with rio.open(out_path, 'w', height=img.shape[0],
                                                  width=img.shape[1], count=2,
                                                  dtype=rio.float32) as dst:
                                        dst.write(img['r'].astype(np.float32), 1)
                                        dst.write(img['i'].astype(np.float32), 2)
                                elif product == 'GSLC':
                                    with rio.open(out_path, 'w', height=img.shape[0],
                                                  width=img.shape[1], count=1,
                                                  dtype=rio.float32) as dst:
                                        dst.write(img.astype(np.float32), 1)
        return combination

    def get_identification_data(self, rslc_file):
        """
        Extract satellite pass identification metadata.

        Returns:
            dict: Keys include trackNumber, frameNumber, processingDateTime,
                  boundingPolygon, and other identification fields.
        """
        output_dict = {}
        rslc_h5 = hp.File(rslc_file, 'r')
        for folder in self.folders:
            if folder not in rslc_h5: continue
            for sar in self.sars:
                if sar not in rslc_h5[folder]: continue
                id_data = rslc_h5[folder][sar]['identification']
                for key in id_data.keys():
                    val = id_data[key]
                    if val.shape == ():
                        output_dict[key] = val[()].decode() if isinstance(val[()], bytes) else val[()]
                    else:
                        output_dict[key] = [x.decode() if isinstance(x, bytes) else x for x in val[:]]
        return output_dict

    def generate_calibrated_image(self, rslc_file, parameter='sigma0',
                                  polarization='HH', interpolation_grid_size=1000):
        """
        Generate a radiometrically calibrated SAR backscatter image.

        Applies calibration LUT via bilinear interpolation over the full scene.

        Args:
            rslc_file (str): Path to NISAR HDF5 file.
            parameter (str): Calibration target — 'sigma0', 'beta0', or 'gamma0'.
            polarization (str): Polarization channel to calibrate.
            interpolation_grid_size (int): Block size for LUT interpolation.

        Returns:
            np.ndarray: Calibrated backscatter image array.
        """
        assert parameter in ['beta0', 'sigma0', 'gamma0'], \
            f"Unsupported parameter '{parameter}'. Choose: beta0, sigma0, gamma0."

        rslc_h5 = hp.File(rslc_file, 'r')
        combos = self.get_tiff_combinations(rslc_file)
        available_pols = [x['datapath'].split('/')[-1] for x in combos]
        assert polarization in available_pols, \
            f"Polarization '{polarization}' not available. Found: {available_pols}"

        datapath = next(c['datapath'] for c in combos
                        if c['datapath'].split('/')[-1] == polarization)

        HH_data = rslc_h5[datapath][()]
        s0_lut = self.get_cal_info(rslc_file)['calibrationInformation'][parameter]
        product_level = self.get_identification_data(rslc_file)['productType']
        assert product_level in ['RSLC', 'GSLC']

        x_factor = HH_data.shape[0] / s0_lut.shape[0]
        y_factor = HH_data.shape[1] / s0_lut.shape[1]
        lut_x = np.arange(s0_lut.shape[0]) * x_factor
        lut_y = np.arange(s0_lut.shape[1]) * y_factor
        lut_interp = RegularGridInterpolator((lut_x, lut_y), s0_lut,
                                             fill_value=None, bounds_error=False)
        calibrated = np.zeros(HH_data.shape)
        gs = interpolation_grid_size

        def _calibrate_block(data_block, i0, j0):
            Y, X = np.meshgrid(
                np.arange(j0, j0 + data_block.shape[1]),
                np.arange(i0, i0 + data_block.shape[0])
            )
            lut_vals = lut_interp((X, Y)) ** 2
            if product_level == 'RSLC':
                return np.abs(data_block['r'] + 1j * data_block['i']) ** 2 / lut_vals
            else:
                return np.abs(data_block) ** 2 / lut_vals

        for i in tqdm(range(0, HH_data.shape[0], gs), desc=f"Calibrating {parameter}"):
            for j in range(0, HH_data.shape[1], gs):
                block = HH_data[i:i+gs, j:j+gs]
                calibrated[i:i+gs, j:j+gs] = _calibrate_block(block, i, j)

        return calibrated

    def retrieve_element(self, grp_name, obj):
        """Helper: recursively extract scalar HDF5 values into cal_dict."""
        self.cal_dict[obj.name.split('/')[-1]] = (
            obj[()].decode() if isinstance(obj[()], bytes) else obj[()]
        )

    def get_cal_info(self, rslc_file):
        """Extract full calibration information blocks from NISAR HDF5."""
        meta_dict = {}
        rslc_h5 = hp.File(rslc_file, 'r')
        for folder in self.folders:
            if folder not in rslc_h5: continue
            for sar in self.sars:
                if sar not in rslc_h5[folder]: continue
                for product in self.products:
                    if product not in rslc_h5[folder][sar]: continue
                    prod_data = rslc_h5[folder][sar][product]
                    if 'metadata' not in prod_data: continue
                    for subgroup in prod_data['metadata'].keys():
                        self.cal_dict = {}
                        prod_data[f'metadata/{subgroup}'].visititems(self.retrieve_element)
                        meta_dict[subgroup] = self.cal_dict
        return meta_dict
