#!/bin/bash

rm -Rfv polynomials.json prompt.txt results.csv
find . -type d -name '__pycache__' -print0 | xargs -0 rm -Rfv
