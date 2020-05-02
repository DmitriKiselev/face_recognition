#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""
Template doc
"""
import argparse


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save-to',
                        help='Path to save dir.')

    return parser.parse_args()


def main():
    """Application entry point."""
    args = get_args()


if __name__ == '__main__':
    main()
