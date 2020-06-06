# Include standard modules
import argparse

def parseArguments():
    # Initiate the parser

    parser = argparse.ArgumentParser()a
    parser.add_argument("-V", "--version", help="show program version", action="store_true")
    parser.add_argument("--model", "-m", help = "Select model type")
    parser.add_argument("--train", "-t", help = "Train the model")
    parser.add_argument("--eval", "-e", help = "Evaluate the model")
    parser.add_argument("--use", "-u", help = "Use the model")
    # Read arguments from the command line
    args = parser.parse_args()

    # Check for --version or -V
    if args.version:
        print("This is myprogram version 0.1")

    if args.train:
        print("Model is %s" % args.train)


parseArguments()