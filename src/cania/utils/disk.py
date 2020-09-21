from pathlib import Path
import pandas as pd

from cgpip.utils.image import read_rgb, write_rgb, read_tiff

PNG = '.png'
CSV = '.csv'
TIF = '.tif'


class Disk(object):
    def __init__(self, location):
        self.location = Path(location)
        if not self.location.exists():
            raise FileNotFoundError()

    def write(self, filename, filedata):
        filepath = self.location / filename
        extension = filepath.suffix
        filepath = str(filepath)
        if extension == PNG:
            return write_rgb(filepath, filedata)

    def read(self, filename):
        filepath = self.location / filename
        extension = filepath.suffix
        filepath = str(filepath)
        if extension == PNG:
            return read_rgb(filepath)
        if extension == CSV:
            return pd.read_csv(filepath)
        if extension == TIF:
            return read_tiff(filepath)

    def next(self, next_location):
        filepath = self.location / next_location
        filepath.mkdir(parents=True, exist_ok=True)
        return Disk(filepath)

    def ls(self, regex='*'):
        return self.location.glob(regex)
