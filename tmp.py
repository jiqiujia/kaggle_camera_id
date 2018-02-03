from numpy import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fun', '-f', default='ones', type=str)
args = parser.parse_args()

print(globals()[args.fun](2))