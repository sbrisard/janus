#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Janus documentation build configuration file, created by
# sphinx-quickstart on Tue Sep 17 10:23:19 2013.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.doctest',
              'sphinx.ext.todo',
              'sphinx.ext.coverage',
              'sphinx.ext.mathjax',
              'sphinx.ext.viewcode']

todo_include_todos = True

import sphinx
if sphinx.__version__ < '1.3':
    extensions.append('sphinxcontrib.napoleon')
else:
    extensions.append('sphinx.ext.napoleon')

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'

project = 'Janus'

with open('../LICENSE.txt', 'r') as f:
    copyright = next(f).lstrip('Copyright (c)').strip()

version = '0.0'
release = '0.0'

exclude_patterns = ['_build']

pygments_style = 'sphinx'

numfig = True

html_logo = './logo_janus-200x200.png'
html_static_path = ['_static']
htmlhelp_basename = 'janusdoc'

latex_elements = {'preamble': '',
                  'inputenc': '',
                  'fontenc': '',
                  'utf8extra': ''}
latex_documents = [
  ('index', 'janus.tex', 'Documentation of the Janus Library',
   'S. Brisard', 'manual'),
]
latex_logo = './logo_janus.png'

man_pages = [
    ('index', 'janus', 'Documentation of the Janus Library',
     ['S. Brisard'], 1)
]

texinfo_documents = [
  ('index', 'janus', 'Documentation of the Janus Library',
   'S. Brisard', 'Janus', 'One line description of project.',
   'Miscellaneous'),
]

autodoc_member_order = 'groupwise'
