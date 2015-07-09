# -*- coding: utf-8 -*-

import distutils.command.clean
import os
import re
import sys
import setuptools

import numpy

from argparse import ArgumentParser
from configparser import ConfigParser
from configparser import NoOptionError

from distutils.core import setup, Command
from distutils.extension import Extension
from distutils.sysconfig import get_config_var
from distutils.util import get_platform
from distutils.dir_util import remove_tree
from distutils import log

from Cython.Build import cythonize

NAME = 'Janus'
DESCRIPTION = ''
LONG_DESCRIPTION = ''
AUTHOR = 'S. Brisard'
AUTHOR_EMAIL = 'sebastien.brisard@ifsttar.fr'
URL = 'https://bitbucket.org/sbrisard/janus/'
DOWNLOAD_URL = 'https://bitbucket.org/sbrisard/janus/'
LICENSE = 'BSD-3'

class clean(distutils.command.clean.clean):
    description = (distutils.command.clean.clean.description +
                   ', including *.c, *.pyc, *.pyd, *.pyo and *.so files')

    def find_directories_to_remove(self, root):
        directories = []
        for dirpath, dirnames, filenames in os.walk(root):
            for dirname in dirnames:
                if dirname == '__pycache__':
                    directories.append(os.path.join(dirpath, dirname))
        return directories

    def find_files_to_remove(self, root):
        p = re.compile('.+\.((c)|(so)|(pyc)|(pyd)|(pyo))$')
        files = []
        for dirpath, dirnames, filenames in os.walk(root):
            if not dirpath.endswith('__pycache__'):
                for filename in filenames:
                    if p.match(filename):
                        files.append(os.path.join(dirpath, filename))
        return files

    def remove_directories(self, directories):
        for d in directories:
            remove_tree(d, dry_run=self.dry_run)

    def remove_files(self, files):
        for f in files:
            log.info('removing '+f)
            if not self.dry_run:
                os.remove(f)

    def run(self):
        out = super().run()

        root = os.path.join('.', 'janus')
        directories = self.find_directories_to_remove(root)
        files = self.find_files_to_remove(root)
        self.remove_directories(directories)
        self.remove_files(files)
        return out

include_dirs = [numpy.get_include()]
library_dirs = []

parser = ArgumentParser(add_help=False)
parser.add_argument('--config')
args, unknown = parser.parse_known_args()
sys.argv = [sys.argv[0]] + unknown
config = args.config

parser = ConfigParser()
parser.read('janus.cfg')
if config is not None:
    fftw3 = parser.get(config, 'fftw3')
    include_dirs.append(parser.get(config, 'fftw3-include'))
    library_dirs.append(parser.get(config, 'fftw3-library'))
    with_mpi = parser.getboolean(config, 'with-mpi')
    if with_mpi:
        fftw3_mpi = parser.get(config, 'fftw3_mpi')
        mpicc = parser.get(config, 'mpicc')
else:
    fftw3 = 'fftw3'
    fftw3_mpi = 'fftw3_mpi'
    with_mpi = True
    mpicc = '/usr/bin/mpicc'

extensions = []

extensions.append(Extension('janus.utils.checkarray',
                            sources=['janus/utils/checkarray.pyx']))

extensions.append(Extension('janus.operators',
                            sources=['janus/operators.pyx']))

extensions.append(Extension('janus.material.elastic.linear.isotropic',
                            sources=['janus/material/elastic/linear/isotropic.pyx']))

extensions.append(Extension('janus.green',
                            sources=['janus/green.pyx'],
                            include_dirs=include_dirs,
                            library_dirs=library_dirs))

# TODO This module also depends on fftw.pxd
extensions.append(Extension('janus.fft.serial._serial_fft',
                            sources=['janus/fft/serial/_serial_fft.pyx'],
                            libraries=[fftw3],
                            library_dirs=library_dirs,
                            include_dirs=include_dirs))

if with_mpi:
    import subprocess

    import mpi4py
    include_dirs.append(mpi4py.get_include())

    showme_compile = subprocess.check_output([mpicc, '--showme:compile']).decode('ascii')
    showme_link = subprocess.check_output([mpicc, '--showme:link']).decode('ascii')

    # TODO This module also depends on fftw_mpi.pxd
    extensions.append(Extension('janus.fft.parallel._parallel_fft',
                                sources=['janus/fft/parallel/_parallel_fft.pyx'],
                                libraries=[fftw3, fftw3_mpi],
                                library_dirs=library_dirs,
                                include_dirs=include_dirs,
                                extra_compile_args=showme_compile.split(),
                                extra_link_args=showme_link.split()))

packages = ['janus',
            'janus.fft',
            'janus.fft.serial',
            'janus.utils']

if with_mpi:
    packages.append('janus.fft.parallel')

setup(name=NAME,
      version='0.1',
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      url=URL,
      download_url=DOWNLOAD_URL,
      license=LICENSE,
      packages=packages,
      ext_modules=cythonize(extensions,
                            compiler_directives={'embedsignature': True}),
      cmdclass={'clean': clean})
