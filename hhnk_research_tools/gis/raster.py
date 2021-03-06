# %%
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage
import os

class Raster:
    def __init__(self, source_path, min_block_size=1024):
        self.source_path = source_path
        self.source_set=False #Tracks if the source exist on the system. 
        self.source = self.source_path #calls self.source.setter(source_path)

        self._array = None
        self.min_block_size=min_block_size

    @property
    def array(self):
        if self._array  is not None:
            print('from memory')
            return self._array
        else:
            print('Array not loaded. Call Raster.get_array(window) first')
            return self._array

    @array.setter
    def array(self, raster_array, window=None, band_nr=1):
        self._array = raster_array
        # self.source.GetRasterBand(band_nr).WriteArray(window[0],window[1],raster_array)

    def _read_array(self, band=None, window=None):
        """window=[x0, y0, x1, y1]--oud.
        window=[x0, y0, xsize, ysize]"""
        if band == None:
            band = self.source.GetRasterBand(1) #TODO band is not closed properly

        if window is not None:
            # raster_array = band.ReadAsArray(
            #     xoff=int(window[0]),
            #     yoff=int(window[1]),
            #     win_xsize=int(window[2] - window[0]),
            #     win_ysize=int(window[3] - window[1]))

            raster_array = band.ReadAsArray(
                xoff=window[0],
                yoff=window[1],
                win_xsize=window[2],
                win_ysize=window[3])
        else:
            raster_array = band.ReadAsArray()

        band.FlushCache()  # close file after writing
        band = None

        return raster_array

    def get_array(self, window=None):
        try:
            if self.band_count == 1:
                raster_array = self._read_array(band=self.source.GetRasterBand(1), 
                                                window=window)

            elif self.band_count == 3:
                red_array = self._read_array(band=self.source.GetRasterBand(1), 
                                                window=window)
                green_array = self._read_array(band=self.source.GetRasterBand(2), 
                                                window=window)   
                blue_array = self._read_array(band=self.source.GetRasterBand(3), 
                                window=window)                                                                            

                raster_array = np.dstack((red_array, green_array, blue_array))
            else:
                raise ValueError(
                    f"Unexpected number of bands in raster {self.source_path} (expect 1 or 3)"
                )
            self._array = raster_array
            return raster_array

        except Exception as e:
            raise e from None


    @property
    def source(self):
        if not self.source_set:
            self.source=self.source_path
            return self._source
        else:
            return self._source
    @source.setter
    def source(self, value):
        """If source does not exist it will not be set.
        Bit clunky. But it should work that if it exists it will only be set once. 
        Otherwise it will not set.  """
        if os.path.exists(self.source_path): #cannot use self.exists here.
            #Needs to be first otherwise we end in a loop when settings metadata/nodata/band_count
            self.source_set=True 

            self._source=gdal.Open(value)
            self.metadata = True #Calls self.metadata.setter
            self.nodata=True
            # self.band_count=self.source.RasterCount
            


    @property
    def exists(self):
        if self.source_set: #check this first for speed.
            return True
        else:
            path_exists = os.path.exists(self.source_path)
            if not self.source_set:
                if path_exists:
                    self.source #Set the source.      
            return path_exists


    @property
    def nodata(self):
        if self.exists:
            return self._nodata

    @nodata.setter
    def nodata(self, val) -> dict:
        self._nodata = self.source.GetRasterBand(1).GetNoDataValue()

    @property
    def band_count(self):
        if self.exists:
            return self.source.RasterCount

    @property
    def metadata(self):
        if self.exists:
            return self._metadata
    @metadata.setter
    def metadata(self, val) -> dict:

            meta = {}
            meta["proj"] = self.source.GetProjection()
            meta["georef"] = self.source.GetGeoTransform()
            meta["pixel_width"] = meta["georef"][1]
            meta["x_min"] = meta["georef"][0]
            meta["y_max"] = meta["georef"][3]
            meta["x_max"] = meta["x_min"] + meta["georef"][1] * self.source.RasterXSize
            meta["y_min"] = meta["y_max"] + meta["georef"][5] * self.source.RasterYSize
            meta["bounds"] = [meta["x_min"], meta["x_max"], meta["y_min"], meta["y_max"]]
            # for use in threedi_scenario_downloader
            meta["bounds_dl"] = {
                "west": meta["x_min"],
                "south": meta["y_min"],
                "east": meta["x_max"],
                "north": meta["y_max"],
            }
            meta["x_res"] = self.source.RasterXSize
            meta["y_res"] = self.source.RasterYSize
            meta["shape"] = [meta["y_res"], meta["x_res"]]

            self._metadata = meta

    def plot(self):
        plt.imshow(self._array)


    @property
    def shape(self):
        return self.metadata['shape']


    @property
    def pixelarea(self):
        return abs(self.metadata['georef'][1] * self.metadata['georef'][5])


    def generate_blocks(self):
        """Generate blocks with the blocksize of the band. 
        These blocks can be used as window to load the raster iteratively."""
        band = self.source.GetRasterBand(1)

        block_height, block_width = band.GetBlockSize()

        if (block_height < self.min_block_size) or (block_width < self.min_block_size):
            block_height=self.min_block_size
            block_width=self.min_block_size

        ncols = int(np.floor(band.XSize / block_width))
        nrows = int(np.floor(band.YSize / block_height))


        #Create arrays with index of where windows end. These are square blocks. 
        xparts = np.linspace(0, block_width*ncols, ncols+1).astype(int)
        yparts = np.linspace(0, block_height*nrows, nrows+1).astype(int)

        #If raster has some extra data that didnt fall within a block it is added to the parts here.
        #These blocks are not square.
        if block_width*ncols != self.shape[1]:
            xparts=np.append(xparts, self.shape[1])
            ncols+=1
        if block_height*nrows != self.shape[0]:
            yparts=np.append(yparts, self.shape[0])
            nrows+=1

        blocks_df = pd.DataFrame(index=np.arange(nrows*ncols)+1, columns=['ix', 'iy', 'window'])
        i = 0
        for ix in range(ncols):
            for iy in range(nrows):
                i += 1
                blocks_df.loc[i, :] = np.array((ix, iy, [xparts[ix], yparts[iy], xparts[ix+1], yparts[iy+1]]), dtype=object)

        blocks_df['window_readarray'] = blocks_df['window'].apply(lambda x: [int(x[0]), int(x[1]), int(x[2]-x[0]), int(x[3]-x[1])])

        band.FlushCache()  # close file after writing
        band = None

        self.blocks=blocks_df
        return blocks_df


    def sum_labels(self, labels_raster, labels_index):
        """Calculate the sum of the rastervalues per label."""
        if labels_raster.shape != self.shape:
            raise Exception(f'label raster shape {labels_raster.shape} does not match the raster shape {self.shape}')

        accum = None

        for window, block in self:
            block[block==self.nodata] = 0
            block_label = labels_raster._read_array(window=window)

            #Calculate sum per label (region)
            result = ndimage.sum_labels(input=block,
                                    labels=block_label,
                                    index=labels_index) #Which values in labels to take into account.

            if accum is None:
                accum = result
            else:
                accum += result
        return accum

    def to_file():
        pass

    def __iter__(self):
        if not hasattr(self,'blocks'):
            _ = self.generate_blocks()

        for idx, block_row in self.blocks.iterrows():
            window=block_row['window_readarray']
            block = self._read_array(window=window)
            yield window, block
            

    def __repr__(self):
        if self.exists:
            return f"""{self.__class__}
    Source: {self.source_path}, exists:{self.exists}
    Shape: {self.metadata['shape']}
    Pixelsize: {self.metadata['pixel_width']}"""
        else:
            return f"""{self.__class__}
    Source: {self.source_path}, exists:{self.exists}"""

if __name__ == '__main__':
    dem_path = r"G:\02_Werkplaatsen\06_HYD\Projecten\HKC16015 Wateropgave 2.0\11. DCMB\hhnk-modelbuilder-master\data\fixed_data\DEM\DEM_AHN4_int.vrt"
    r=Raster(dem_path)
    print(r)

# %%
