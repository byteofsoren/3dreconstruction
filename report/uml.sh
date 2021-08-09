#!/bin/bash

# ==== Uml.sh ====
# This script generates the nesssesary uml files for the report
# Written by Magnus Sörensen
# ===============

# How to use:

umlDir="figures/plantuml"
target=$(pwd)"/figures/uml"

currentDir=$(pwd)
cd $umlDir
# Create target dir if not found
if [[ ! -d $target ]]; then
    echo target not found create directory
    mkdir $target
fi
find . -type f -name '*.uml' | while read fp
do
    # echo $fp
    fname=$(echo $fp | sed 's/^\.\///g' | cut -f 1 -d '.')
    echo "Converting $fname.uml to eps"
    plantuml $fname.uml -eps
    mv $fname.eps $target/.
    if [[ -f $target/$fname.eps ]]; then
        echo "[OK] Convert to pdf"
        epstopdf $target/$fname.eps
        rm $target/$fname.eps
    else
        echo "[ERROR] $target/$fname.eps not found"
    fi
done
cd $currentDir
