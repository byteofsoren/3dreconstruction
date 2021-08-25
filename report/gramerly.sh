#!/bin/bash

# This program copies the files in the
folder="sections"
ftype="*.tex"
# Directory to the clip board
# After that you will get a message to paste
# the text in to grammerly
# then after touching up the text in gramerly
# you press [p] to paste back the text in to
# the same file again.

find $folder -type f -name "$ftype" | while read fp; do
    echo $fp
done

