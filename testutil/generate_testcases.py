#!/usr/bin/env python
import sys
import json
import random
import logging
import argparse

# expect integer input specifying the length of the generated array
# output builder is evaluated at order 0 (first)
def length_constraint(length):
    _default_length = 8
    if length is None or type(length) is not int:
        length = _default_length
    return 0, lambda _: range(length)

# expect integer input specifying the maximum np.prod of generated array
# output builder is evaluated last
def nprod_constraint(max_nprod):
    _default_max_nprod = int('ffff', 16)
    if max_nprod is None or type(max_nprod) is not int:
        max_nprod = _default_max_nprod
    def ret_func(arr):
        out = []
        acc = 1
        for e in arr:
            acc *= e
            if acc > max_nprod:
                return out
            out.append(e)
        return out
    return 2, ret_func

# cfg has form {"min": <int>, "max": <int>} representing the
# potential interval of each value in generated array
# output builder is evaluated at order 0 (second)
def val_range_constraint(cfg):
    _default_minper = 1
    _default_maxper = 255
    if cfg is not None:
        min_perdim = cfg.get('min', _default_minper)
        max_perdim = cfg.get('max', _default_maxper)
    else:
        min_perdim = _default_minper
        max_perdim = _default_maxper
    return 1, lambda arr: [random.randint(min_perdim, max_perdim) for _ in arr]

_default_outfile = '/tmp/testcases.json'

# all constraints values are lambdas that:
# return order (integer), and builder (lambda input array return constraint array)
_default_constraints = {
    'length': length_constraint,
    'nprod': nprod_constraint,
    'val_range': val_range_constraint,
}

def generate_testcases(test_template, constraint_cfgs,
    constraints = _default_constraints):

    # type check
    assert type(test_template) is dict
    assert type(constraint_cfgs) is dict

    cfg_builders = {}
    for cfg_type in constraint_cfgs:
        cfg = constraint_cfgs[cfg_type]
        constraint_args = dict([(key, None) for key in constraints])
        for ctype in cfg:
            if ctype in constraint_args:
                constraint_args[ctype] = cfg[ctype]
        logging.info('parsing constraint `{}`: {}'.
            format(cfg_type, constraint_args))
        funcs = []
        for ctype in constraint_args:
            funcs.append(constraints[ctype](constraint_args[ctype]))
        funcs.sort(key=lambda x: x[0])
        builders = [func for _, func in funcs]
        cfg_builders[cfg_type] = builders

    outdata = {}
    for func_ctx in test_template:
        config_types = test_template[func_ctx]
        outdata[func_ctx] = []
        for cfg_type in config_types:
            if cfg_type not in cfg_builders:
                continue
            builders = cfg_builders[cfg_type]
            logging.info('generating for context `{}` with constraint `{}`'.
                format(func_ctx, cfg_type))
            arr = []
            for builder in builders:
                arr = builder(arr)
            logging.debug('generated {}'.format(arr))
            outdata[func_ctx].append(arr)
    return outdata

def validate_loglevel(opt):
    if opt == 'info':
        return logging.INFO
    elif opt == 'debug':
        return logging.DEBUG
    elif opt == 'warning':
        return logging.WARNING
    elif opt == 'error':
        return logging.ERROR
    raise argparse.ArgumentTypeError(
        'log levels expected. [info/debug/warning/error]')

def main():
    _prog_description = 'Generate testcases give testcase templates'

    parser = argparse.ArgumentParser(description=_prog_description)
    parser.add_argument('--temp_file', dest='template_file', required=True,
        help='Filename to load template json file from')
    parser.add_argument('--out', dest='outfile', nargs='?',
        default='/tmp/testcases.json',
        help='Filename to write generated testcases '+
            '(default: /tmp/testcases.json)')
    parser.add_argument('--log_level', dest='log_level',
        type=validate_loglevel, default='debug',
        help='Log level (default: debug)')
    args = parser.parse_args()

    outfile = args.outfile
    template_file = args.template_file

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

    with open(template_file) as json_data:
        test_template = json.load(json_data)
        assert 'test_cases' in test_template
        assert 'config_pools' in test_template

    outdata = generate_testcases(
        test_template['test_cases'],
        test_template['config_pools'])

    with open(outfile, 'w') as out:
        out.write(json.dumps(outdata))

if __name__ == '__main__':
    main()
