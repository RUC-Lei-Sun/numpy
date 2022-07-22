import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--name', help="helllo")

print(parser.parse_args())