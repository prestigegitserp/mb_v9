#!/usr/bin/env python3
"""Versioned entry point; explicit legacy entries retain their original behavior."""
import argparse


def run(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--code-version", choices=("2", "3", "4", "5"), default="5")
    options, remaining = parser.parse_known_args(argv)
    if options.code_version == "2":
        from v2.entry import run
    elif options.code_version == "3":
        from v3.entry import run
    elif options.code_version == "4":
        from v4.entry import run
    else:
        from v5.entry import run
    return run(remaining)


if __name__ == "__main__":
    run()
