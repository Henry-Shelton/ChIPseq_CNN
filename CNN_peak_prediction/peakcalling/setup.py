from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

cython_ext_modules = [Extension("readCounts",["readCounts.pyx"],
                                include_dirs=[numpy.get_include()])
                      ,
            Extension("bedGen",["genBedRows.pyx"],
                      include_dirs=[numpy.get_include()])]

c_ext_modules =[Extension("readbam",
                          sources = [
                          "./bamdepth/readbam.c",
                          "./bamdepth/htslib_1_9/libhts.a"
                          ],
                          include_dirs=[
                          "./bamdepth/htslib_1_9",
                          "./bamdepth/htslib_1_9/htslib",
                          numpy.get_include()
                          ],
                          library_dirs=[
                          "./bamdepth/htslib_1_9"
                          ,"./bamdepth/htslib_1_9/htslib"
                          ],
                          libraries=["z","m","pthread"])]


setup(
        ext_modules = cythonize(cython_ext_modules)
    )