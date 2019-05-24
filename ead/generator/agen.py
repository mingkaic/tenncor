''' Generate glue layer '''

import argparse
import yaml
import os.path
import sys
import logging

from ead.generator.plugins.apis import APIsPlugin
from ead.generator.plugins.dtypes import DTypesPlugin
from ead.generator.plugins.opcodes import OpcodesPlugin

from gen.generate import generate

prog_description = 'Generate c++ glue layer mapping ADE and some data-processing library.'

def parse(cfg_str):
    args = yaml.safe_load(cfg_str)
    if type(args) != dict:
        raise Exception('cannot parse non-root object {}'.format(cfg_str))
    return args

def main(args):

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

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

    print(OpcodesPlugin)
    print(DTypesPlugin)
    print(APIsPlugin)
    generate(fields, outpath=outpath,
        plugins=[OpcodesPlugin(), DTypesPlugin(), APIsPlugin()])

if '__main__' == __name__:
    main(sys.argv[1:])
