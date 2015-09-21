import math

import pyx


HALF_SQRT_3 = 0.8660254037844386

DIM_LEG = 0.15

def point_3D(x, y, z):
    return HALF_SQRT_3*(-x+y), -0.5*(x+y)+z


def multiline(points, closed=False):
    it = iter(points)
    items = [pyx.path.moveto(*next(it))]
    for point in it:
        items.append(pyx.path.lineto(*point))
    if closed:
        items.append(pyx.path.closepath())
    return pyx.path.path(*items)

def pgfnode(x, y, text, anchor, use, f):
    f.write('\\begin{pgfscope}\n')
    f.write('\\pgftransformshift{{\\pgfpoint{{{}cm}}{{{}cm}}}}\n'.format(x, y))
    f.write('\\pgfnode{{rectangle}}{{{0}}}{{{1}}}'
            '{{}}{{\\pgfusepath{{{2}}}}}\n'.format(anchor, text, use))
    f.write('\\end{pgfscope}\n')

def dim_legs_3D(x, y, z):

    items = []
    items.append(pyx.path.moveto(*point_3D(x-DIM_LEG, y, z)))
    items.append(pyx.path.lineto(*point_3D(x+DIM_LEG, y, z)))

    items.append(pyx.path.moveto(*point_3D(x, y-DIM_LEG, z)))
    items.append(pyx.path.lineto(*point_3D(x, y+DIM_LEG, z)))

    items.append(pyx.path.moveto(*point_3D(x, y, z-DIM_LEG)))
    items.append(pyx.path.lineto(*point_3D(x, y, z+DIM_LEG)))

    return pyx.path.path(*items)

def dim_3D(x1, y1, z1, x2, y2, z2, canvas):
    linewidth = pyx.style.linewidth.thin
    X1, Y1 = point_3D(x1, y1, z1)
    X2, Y2 = point_3D(x2, y2, z2)
    canvas.stroke(pyx.path.line(X1, Y1, X2, Y2), [linewidth,
                                                  pyx.deco.barrow(size=0.15),
                                                  pyx.deco.earrow(size=0.15)])
    canvas.stroke(dim_legs_3D(x1, y1, z1), [linewidth])
    canvas.stroke(dim_legs_3D(x2, y2, z2), [linewidth])

def dim_legs(x1, y1, x2, y2):
    dx = x2-x1
    dy = y2-y1
    ds = math.sqrt(dx**2+dy**2)
    c = dx/ds*DIM_LEG
    s = dy/ds*DIM_LEG
    items = [pyx.path.moveto(x1, y1),
             pyx.path.lineto(x1-c, y1-s),
             pyx.path.moveto(x1+s, y1-c),
             pyx.path.lineto(x1-s, y1+c),
             pyx.path.moveto(x2, y2),
             pyx.path.lineto(x2+c, y2+s),
             pyx.path.moveto(x2+s, y2-c),
             pyx.path.lineto(x2-s, y2+c)]
    return items

def dim(x1, y1, x2, y2, canvas):
    attrs = [pyx.deco.stroked([pyx.style.linewidth.thin])]
    canvas.stroke(pyx.path.path(*dim_legs(x1, y1, x2, y2)), attrs)
    attrs += [pyx.deco.barrow([pyx.deco.filled([pyx.color.gray.black])],
                              size=0.15),
              pyx.deco.earrow([pyx.deco.filled([pyx.color.gray.black])],
                              size=0.15)]
    canvas.stroke(pyx.path.line(x1, y1, x2, y2), attrs)

def boxed_text(x, y, text, text_attrs, box_attrs, canvas, inner_sep=0.05):
    text = pyx.text.text(x, y, text, text_attrs)
    box = text.bbox().copy()
    box.enlarge(inner_sep)
    canvas.draw(box.path(), box_attrs)
    canvas.insert(text)
