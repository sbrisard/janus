# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first three.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
PYTHON        ?= python
SOURCEDIR     = .
BUILDDIR      = _build
# The HTML docs published on GitHub Pages (https://sbrisard.github.io/janus/)
# are served from the docs/ directory at the root of the project.
GHPAGESDIR    = ../docs

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo
	@echo "Janus-specific target:"
	@echo "  ghpages     to empty ../docs (except .nojekyll) and build the HTML docs"
	@echo "              for GitHub Pages into it"

.PHONY: help ghpages Makefile

# Empty the docs/ directory (except docs/.nojekyll), then build the HTML docs
# into it from scratch.
ghpages:
	@$(PYTHON) ../scripts/empty_docs.py
	@$(SPHINXBUILD) -b html -E -d "$(BUILDDIR)/doctrees" "$(SOURCEDIR)" "$(GHPAGESDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
