"""Empty the docs/ directory, prior to building the HTML docs for GitHub Pages.

All files and subdirectories of docs/ are removed, except docs/.nojekyll
(which tells GitHub Pages not to process the site with Jekyll). This ensures
that files that Sphinx no longer produces do not linger in docs/.

The docs/ directory is located relative to this script (at the root of the
project), not relative to the current directory. The HTML docs are then built
with (from the root of the project)

    python -m sphinx -b html -E -d sphinx/_build/doctrees sphinx docs

"""

import pathlib
import shutil

KEEP = {'.nojekyll'}


def empty_docs(docs):
    if not docs.is_dir():
        return
    for path in docs.iterdir():
        if path.name in KEEP:
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


if __name__ == '__main__':
    docs = pathlib.Path(__file__).resolve().parent.parent / 'docs'
    empty_docs(docs)
    print('Emptied {0} (kept {1})'.format(docs, ', '.join(sorted(KEEP))))
