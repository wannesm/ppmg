#!/usr/bin/env python3
# encoding: utf-8
"""
"""
import sys
import logging


logger = logging.getLogger("be.kuleuven.cs.dtai.ppmg")


try:
    from ppmg import cli
except ModuleNotFoundError:
    logger.info('Cannot find ppmg package. Looking for the "src" directory.')
    sys.path.append('src')
    from ppmg import cli

if __name__ == "__main__":
    sys.exit(cli.main())
