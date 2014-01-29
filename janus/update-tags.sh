#!/bin/bash
find . -type f -iname "*.py" -o -iname "*.pyx" -o -iname "*.pxd" | etags -
