"""Setup xorca."""

from setuptools import setup

setup(name='xorca',
      description='Work on the ORCA grid with XGCM and Xarray',
      packages=['xorca'],
      package_dir={'xorca': 'xorca'},
      install_requires=['setuptools', ],
      zip_safe=False)
