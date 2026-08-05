# -*- coding: utf-8 -*-

import configparser
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

# Project metadata lives in pyproject.toml ([project] table).


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


def update_from_config(kwargs, section):
    """Merge the specified section of ``setup.cfg`` into ``kwargs``.

    The ``include_dirs``, ``library_dirs`` and ``libraries`` entries of the
    section are read as comma-separated lists, and appended to the values
    already stored in ``kwargs`` (a dictionary that can be passed to
    Extension()). A missing section, or a missing/empty entry, is ignored.

    ``kwargs`` is updated in place and returned.
    """
    config = configparser.ConfigParser()
    config.read('setup.cfg')
    if config.has_section(section):
        for key in ['include_dirs', 'library_dirs', 'libraries']:
            value = config[section].get(key, '')
            if value != '':
                kwargs[key] = (kwargs.get(key, [])
                               +[token.strip() for token in value.split(',')])
    return kwargs


def extensions_and_packages():
    utils = Extension('janus.utils.checkarray',
                      sources=['janus/utils/checkarray.pyx'])
    operators = Extension('janus.operators',
                          sources=['janus/operators.pyx'])
    materials = Extension('janus.material.elastic.linear.isotropic',
                          sources=['janus/material/elastic/linear/isotropic.pyx'])

    kwargs = update_from_config({}, 'fftw')
    serial_fft = Extension('janus.fft.serial._serial_fft',
                           sources=['janus/fft/serial/_serial_fft.pyx'],
                           **kwargs)
    green = Extension('janus.green', sources=['janus/green.pyx'], **kwargs)
    extensions = [utils, operators, materials, green, serial_fft]
    packages = ['janus', 'janus.fft', 'janus.fft.serial', 'janus.utils']
    return extensions, packages


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
        kwargs = update_from_config(mpicc_show(), 'fftw_mpi')
        parallel_fft = Extension('janus.fft.parallel._parallel_fft',
                                 sources=['janus/fft/parallel/_parallel_fft.pyx'],
                                 **kwargs)
        return [parallel_fft], ['janus.fft.parallel']
    except ImportError:
        return [], []


if __name__ == '__main__':
    extensions, packages = extensions_and_packages()
    extensions_mpi, packages_mpi = extensions_and_packages_with_mpi()

    extensions += extensions_mpi
    packages += packages_mpi
    setup(packages=packages,
          ext_modules=cythonize(extensions,
                                compiler_directives={'embedsignature': True,
                                                     'language_level': 3}),
          cmdclass={'clean': clean})
