''' Generate glue layer '''

import argparse
import json
import os.path
import sys

from age.plugin.internal import InternalPlugin
from age.test.capi_plugin import CAPIPlugin

from gen.generate import generate

prog_description = 'Generate c++ glue layer mapping ADE and some data-processing library.'

def parse(cfg_str):
    args = json.loads(cfg_str)
    if type(args) != dict:
        raise Exception('cannot parse non-root object {}'.format(cfg_str))
    return args

def main(args):

    parser = argparse.ArgumentParser(description=prog_description)
    parser.add_argument('--cfg', dest='cfgpath', nargs='?',
        help='Configuration json file on mapping info (default: read from stdin)')
    parser.add_argument('--out', dest='outpath', nargs='?',
        help='Directory path to dump output files (default: write to stdin)')
    args = parser.parse_args(args)

    cfgpath = args.cfgpath
    if cfgpath:
        with open(str(cfgpath), 'r') as cfg:
            cfg_str = cfg.read()
        if cfg_str == None:
            raise Exception("cannot read from cfg file {}".format(cfgpath))
    else:
        cfg_str = sys.stdin.read()

    fields = parse(cfg_str)
    outpath = args.outpath

    generate(fields, outpath=outpath,
        plugins=[InternalPlugin(), CAPIPlugin()])

if '__main__' == __name__:
    main(sys.argv[1:])
