''' Generate glue layer '''

import argparse
import yaml
import os.path
import sys
import logging

from eteq.gen.plugins.dtypes import DTypesPlugin
from eteq.gen.plugins.opcodes import OpcodesPlugin
from eteq.gen.plugins.apis import APIsPlugin
from eteq.gen.plugins.pyapis import PyAPIsPlugin

from gen.dump import PrintDump, FileDump
from gen.generate import generate

prog_description = 'Generate c++ glue layer mapping TEQ and some data-processing library.'

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
    parser.add_argument('--out', dest='outpath', nargs='?', default='',
        help='Directory path to dump output files (default: write to stdin)')
    parser.add_argument('--strip_prefix', dest='strip_prefix', nargs='?', default='',
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
    strip_prefix = args.strip_prefix

    if len(outpath) > 0:
        includepath = outpath
        if includepath and includepath.startswith(strip_prefix):
            includepath = includepath[len(strip_prefix):].strip("/")
        out = FileDump(outpath, includepath=includepath)
    else:
        out = PrintDump()

    generate(fields, out=out,
        plugins=[DTypesPlugin(), OpcodesPlugin(), APIsPlugin(), PyAPIsPlugin()])

if '__main__' == __name__:
    main(sys.argv[1:])
