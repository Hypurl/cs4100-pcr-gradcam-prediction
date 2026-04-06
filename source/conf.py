# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../pay_attn'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CS4100 Final Project'
copyright = '2026, Christian Garcia, Tommaso Maga, Yu-Chun Ou, Peter SantaLucia'
author = 'Christian Garcia, Tommaso Maga, Yu-Chun Ou, Peter SantaLucia'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # supports Google/NumPy style docstrings
    'sphinx.ext.mathjax',   # renders LaTeX math in HTML output
    'myst_parser',          # allows using Markdown files
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

# -- Options for PDF/LaTeX output --------------------------------------------
plot_formats = [('png', 150)]
