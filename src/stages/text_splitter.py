"""
Dieses Script is für das Einteilen von Text in Texblöcke zuständig.

Usage: script_name.py input_filename output_filename
"""

import argparse
from dill import dump

parser = argparse.ArgumentParser()

#add positional arguments
parser.add_argument('input_filename')
parser.add_argument('output_filename')

#add optional arguments
#parser.add_argument('--extra', '-e')

args = parser.parse_args()

# Access the arguments
input_filename = args.input_filename
output_filename = args.output_filename
#extra = args.extra

#read file content
with open(input_filename, "r") as f:
    text = f.read()

#split by empty lines
blocks = text.split("\n\n")

#remove first block (just source description)
blocks = blocks[1:]

with open(output_filename, "wb") as f:
    dump(blocks, f)


