
template = ('self.g{0}{1} = (' +
            ' + '.join('w{0} * self.g{0}[{{0}}, {{1}}]'.format(i + 1)
                    for i in range(8)) +
            ')')

for i in range(6):
    for j in range(i, 6):
        print(template.format(i, j))

template = 'out[{0}][{1}] = self.g{2}{3}'

for i in range(6):
    for j in range(6):
        print(template.format(i, j, min(i, j), max(i, j)))

def coeff(i, j):
    return 'g{0}{1}'.format(min(i, j), max(i, j))

for i in range(6):
    print('y[{}] = ('.format(i) +
          ' + '.join('self.{0} * x{1}'.format(coeff(i, j), j)
                     for j in range(6)) +
          ')')
