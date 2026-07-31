import sys


def main(argv):
    if len(argv) > 1 and argv[1] == "--greet":
        print("hello")
        return 0
    print("usage: cli.py --greet")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
