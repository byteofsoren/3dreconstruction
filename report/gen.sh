#!/bin/bash
inputname="main.tex"
refname=$(echo $inputname | sed -e 's/\.tex//g')
pdflatex $inputname
pdflatex $inputname
biber $refname
pdflatex $inputname
