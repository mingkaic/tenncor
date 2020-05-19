''' Generate glue layer '''

import argparse
import yaml
import os.path
import sys
import logging
import importlib

from gen.dump import PrintDump, FileDump
from gen.generate import generate

prog_description = 'Generate c++ glue layer mapping TEQ and some data-processing library.'

def parse(cfg_str):
    args = yaml.safe_load(cfg_str)
    if type(args) != dict:
        raise Exception('cannot parse non-root object {}'.format(cfg_str))
    return args

def str2pair(opt):
    pair = opt.split(':')
    if len(pair) != 2:
        raise argparse.ArgumentTypeError('Two values seperated by : expected.')
    return tuple([val.strip() for val in pair])

def main(args):

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

    parser = argparse.ArgumentParser(description=prog_description)
    parser.add_argument('--plugins', dest='plugins', nargs='+', type=str2pair,
        required=True, help='Plugin in the form of <plugin module>:<plugin class>')
    parser.add_argument('--cfgs', dest='cfgpaths', nargs='+',
        help='Configuration yaml files on mapping info (default: read from stdin)')
    parser.add_argument('--ext_path', dest='ext_path', nargs='?', default='',
        help='Configuration yaml path to look for configuration extensions')
    parser.add_argument('--out', dest='outpath', nargs='?', default='',
        help='Directory path to dump output files (default: write to stdin)')
    parser.add_argument('--strip_prefix', dest='strip_prefix', nargs='?', default='',
        help='Directory path to dump output files (default: write to stdin)')
    args = parser.parse_args(args)

    cfgpaths = args.cfgpaths
    if cfgpaths and len(cfgpaths) > 0:
        cfg_strs = []
        for cfgpath in cfgpaths:
            with open(str(cfgpath), 'r') as cfg:
                cfg_str = cfg.read()
            if cfg_str == None:
                raise Exception("cannot read from cfg file {}".format(cfgpath))
            cfg_strs.append(cfg_str)
        cfg_str = '\n'.join(cfg_strs)
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

    plugins = dict(args.plugins)
    generate(fields, out=out, plugins=[
        getattr(importlib.import_module(mod), plugins[mod])()
        for mod in plugins],
        ext_path=args.ext_path)

if '__main__' == __name__:
    main(sys.argv[1:])
