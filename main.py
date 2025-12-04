#!/usr/bin/env python3

import click
import importlib

@click.group()
def main_cli():
    pass

from implementations.sand_move import sand_move
main_cli.add_command(sand_move)

# from implementations.refraccion import refraction
# main_cli.add_command(refraction)

# from implementations.sand_move import sand_move
# main_cli.add_command(sand_move)


if __name__ == '__main__':
    main_cli()