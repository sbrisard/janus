# -*- coding: utf-8 -*-

import distutils.command.clean
import os
import re
import setuptools

from distutils import log
from distutils.core import setup
from distutils.dir_util import remove_tree
from distutils.extension import Extension
from distutils.util import split_quoted

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


def extensions_and_packages():
    extensions = [Extension('janus.utils.checkarray',
                            sources=['janus/utils/checkarray.pyx']),
                  Extension('janus.operators',
                            sources=['janus/operators.pyx']),
                  Extension('janus.material.elastic.linear.isotropic',
                            sources=['janus/material/elastic/linear/'
                                     'isotropic.pyx']),
                  Extension('janus.green',
                            sources=['janus/green.pyx'],),
                  # TODO This module also depends on fftw.pxd
                  Extension('janus.fft.serial._serial_fft',
                            sources=['janus/fft/serial/_serial_fft.pyx'])]
    packages = ['janus', 'janus.fft', 'janus.fft.serial', 'janus.utils']
    return extensions, packages


def mpicc_showme():
    """Use ``mpicc --showme`` to retrieve the mpicc arguments.

    Works with openmpi, not mpich.
    Returns a dictionary that can be passed to Extension().
    """

    import mpi4py
    from subprocess import check_output
    mpicc = mpi4py.get_config()['mpicc']

    def call_mpicc_showme(arg):
        out = check_output([mpicc, '--showme:'+arg])
        return out.decode('ascii').split()

    incdirs = call_mpicc_showme('incdirs')
    incdirs.append(mpi4py.get_include())

    return {'include_dirs': incdirs,
            'library_dirs': call_mpicc_showme('libdirs'),
            'extra_compile_args': call_mpicc_showme('compile'),
            'extra_link_args': call_mpicc_showme('link')}


def mpicc_show():
    """Use ``mpicc --show`` to retrieve the mpicc arguments.

    Works with both openmpi and mpich.
    Returns a dictionary that can be passed to Extension().
    """
    import mpi4py
    import subprocess
    mpicc = mpi4py.get_config()['mpicc']
    mpicc_show = subprocess.check_output([mpicc, '-show']).decode().strip()
    # Strip command line from first part, which is the name of the compiler
    mpicc_show = re.sub('\S+\s', '', mpicc_show, count=1)

    def my_filter(regex, iterable, group=0):
        matching = []
        non_matching = []
        for item in iterable:
            m = re.search(regex, item)
            if m is not None:
                matching.append(m.group(group))
            else:
                non_matching.append(item)
        return matching, non_matching

    cflags = split_quoted(mpicc_show)
    incdirs, cflags = my_filter('^-I(.*)', cflags, 1)
    libdirs, cflags = my_filter('^-L(.*)', cflags, 1)
    ldflags, cflags = my_filter('^-W?l.*', cflags)
    ldflags += cflags
    incdirs.append(mpi4py.get_include())

    return {'include_dirs': incdirs,
            'library_dirs': libdirs,
            'extra_compile_args': cflags,
            'extra_link_args': ldflags}


def extensions_and_packages_with_mpi():
    try:
        # import mpi4py
        # TODO This module also depends on fftw_mpi.pxd
        extensions = [Extension('janus.fft.parallel._parallel_fft',
                                sources=['janus/fft/parallel/'
                                         '_parallel_fft.pyx'],
                                **mpicc_show())]
        packages = ['janus.fft.parallel']
        return extensions, packages
    except ImportError:
        return [], []

if __name__ == '__main__':
    extensions, packages = extensions_and_packages()
    extensions_mpi, packages_mpi = extensions_and_packages_with_mpi()

    extensions += extensions_mpi
    packages += packages_mpi
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
