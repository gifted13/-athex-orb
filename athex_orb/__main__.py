"""python -m athex_orb <εντολή> — βλ. athex_orb/cli.py."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
