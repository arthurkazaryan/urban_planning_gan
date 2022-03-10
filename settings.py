from pathlib import Path


class DataPathData(object):
    base: Path = Path.cwd().joinpath('data')
    arrays: Path = base.joinpath('arrays')
    images: Path = base.joinpath('images')


class MainPathData(object):
    base: Path = Path.cwd()
    data: DataPathData = DataPathData()


PATH_DATA = MainPathData()
