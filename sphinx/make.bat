@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
if "%PYTHON%" == "" (
	set PYTHON=python
)
set SOURCEDIR=.
set BUILDDIR=_build
REM The HTML docs published on GitHub Pages (https://sbrisard.github.io/janus/)
REM are served from the docs/ directory at the root of the project.
set GHPAGESDIR=..\docs

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.https://www.sphinx-doc.org/
	exit /b 1
)

if "%1" == "" goto help
if "%1" == "ghpages" goto ghpages

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:ghpages
REM Empty the docs/ directory (except docs/.nojekyll), then build the HTML docs
REM into it from scratch.
%PYTHON% ..\scripts\empty_docs.py
if errorlevel 1 goto end
%SPHINXBUILD% -b html -E -d %BUILDDIR%\doctrees %SOURCEDIR% %GHPAGESDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
echo.
echo.Janus-specific target:
echo.  ghpages     to empty ..\docs (except .nojekyll) and build the HTML docs
echo.              for GitHub Pages into it

:end
popd
