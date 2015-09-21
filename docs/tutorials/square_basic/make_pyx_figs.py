import os.path

import h5py
import numpy as np
import PIL
import PIL.Image
import pyx

from pyxutil import *

def make_fig_microstructure(name):
    L = 3
    a = 0.75*L
    dim_shift = 0.6

    c = pyx.canvas.canvas()

    attrs = [pyx.style.linewidth.normal, pyx.deco.earrow()]
    c.stroke(pyx.path.line(-DIM_LEG, 0, L+2*DIM_LEG, 0), attrs)
    c.stroke(pyx.path.line(0, -DIM_LEG, 0, L+2*DIM_LEG), attrs)
    c.text(L+2*DIM_LEG, -0.1, r'$x_1$', [pyx.text.halign.boxcenter,
                                         pyx.text.valign.top])
    c.text(-0.1, L+2*DIM_LEG, r'$x_2$', [pyx.text.halign.boxright,
                                         pyx.text.valign.middle])

    attrs = [pyx.deco.filled([pyx.color.gray(0.75)])]
    c.draw(pyx.path.rect(0, 0, L, L), attrs)

    attrs = [pyx.deco.stroked([pyx.style.linewidth.normal]),
             pyx.deco.filled([pyx.color.gray.white])]
    c.draw(pyx.path.rect(0, 0, a, a), attrs)

    attrs = [pyx.deco.stroked([pyx.style.linewidth.Thick])]
    c.draw(pyx.path.rect(0, 0, L, L), attrs)

    dim(0, -dim_shift, a, -dim_shift, c)
    dim(-dim_shift, 0, -dim_shift, a, c)
    dim(0, L+dim_shift, L, L+dim_shift, c)
    dim(L+dim_shift, 0, L+dim_shift, L, c)

    attrs = [pyx.text.halign.boxcenter, pyx.text.valign.middle]
    text = pyx.text.text(0.5*a, -dim_shift, r'\color{black}$a$', attrs)
    c.draw(text.bbox().path(), [pyx.deco.filled([pyx.color.gray.white])])
    c.insert(text)
    text = pyx.text.text(0.5*L, L+dim_shift, r'\color{black}$L$', attrs)
    c.draw(text.bbox().path(), [pyx.deco.filled([pyx.color.gray.white])])
    c.insert(text)

    cc = pyx.canvas.canvas()
    text = pyx.text.text(0, 0, r'\color{black}\color{black}$a$', attrs)
    cc.draw(text.bbox().path(), [pyx.deco.filled([pyx.color.gray.white])])
    cc.insert(text)
    c.insert(cc, [pyx.trafo.rotate(90),
                  pyx.trafo.translate(-dim_shift, 0.5*a)])

    cc = pyx.canvas.canvas()
    text = pyx.text.text(0, 0, r'\color{black}$L$', attrs)
    cc.draw(text.bbox().path(), [pyx.deco.filled([pyx.color.gray.white])])
    cc.insert(text)
    c.insert(cc, [pyx.trafo.rotate(-90),
                  pyx.trafo.translate(L+dim_shift, 0.5*L)])

    c.text(0.5*a, 0.5*a,
           r'\color{black}$\mu_\mathrm{i},\nu_\mathrm{i}$', attrs)
    c.text(0.5*L, 0.5*(L-a)+a,
           r'\color{black}$\mu_\mathrm{m},\nu_\mathrm{m}$', attrs)

    c.writePDFfile(name)
    c.writeSVGfile(name)


if __name__ == '__main__':
    # Using package txfonts leads to LaTeX messages that pyx cannot parse.
    pyx.text.set(pyx.text.LatexRunner,
                 errordetail=pyx.text.errordetail.full,
                 docopt='12pt',
                 texmessages_preamble=[pyx.text.texmessage.ignore],
                 texmessages_run=[pyx.text.texmessage.ignore])
    pyx.text.preamble(r'\usepackage{amsmath, color, txfonts}')

    make_fig_microstructure('microstructure')
