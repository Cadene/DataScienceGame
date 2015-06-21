import argparse
import csv

def main(args):
    f = args.output_file

    lines = open(f,"rb").readlines()




parser = argparse.ArgumentParser()
parser.add_argument("output_file", type=str)
args = parser.parse_args()

main(args)
