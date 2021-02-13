import sys
import json
import os.path
import argparse
import matplotlib.pyplot as plt

from compare_matmul import np_matmul, tc_matmul, tf_matmul
from compare_mlp import tc_mlp, tf_mlp
from compare_mlp_grad import tc_mlp_grad, tf_mlp_grad

matrix_dims = [
    26,
    50,
    76,
    100,
    126,
    150,
    176,
    200,
    226,
    250,
    500,
    1000,
    1500,
]

def main(outdir, bmtype, testname, measure):
    durs = []
    for matrix_dim in matrix_dims:
        dur = measure(matrix_dim)
        durs.append(dur)
    print('{} durations: {}'.format(testname, durs))
    try:
        with open('{}_results.json'.format(bmtype), 'r') as f:
            existing_content = f.read()
            if len(existing_content) == 0:
                existing_content = '{}'
            results = json.loads(existing_content)
    except Exception as e:
        results = {}
    with open(os.path.join(outdir, '{}_results.json'.format(bmtype)), 'w') as f:
        results[testname] = durs
        f.write(json.dumps(results))

def plot_all(np_durs, tc_durs, tf_durs):
    if np_durs is not None:
        np_lines = plt.plot(matrix_dims, np_durs, 'g--', label='numpy durations')
    if tc_durs is not None:
        tc_line = plt.plot(matrix_dims, tc_durs, 'r--', label='tc durations')
    if tf_durs is not None:
        tf_line = plt.plot(matrix_dims, tf_durs, 'b--', label='tf durations')
    plt.legend()
    plt.show()

if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', dest='mode',
        type=str, nargs='?', default='bm',
        help='Either bm or show (default: bm)')
    parser.add_argument('--bmtype', dest='bmtype',
        type=str, nargs='?', default='matmul',
        help='One of [matmul, mlp, mlp_grad] (default: matmul)')
    parser.add_argument('--testname', dest='testname',
        type=str, nargs='?', default='tc',
        help='One of [np, tc, tf] (default: tc)')
    parser.add_argument('--filename', dest='filename',
        type=str, nargs='?', default='',
        help='File to show')
    parser.add_argument('--outdir', dest='outdir',
        type=str, nargs='?', default='',
        help='Output directory')
    args = parser.parse_args(sys.argv[1:])
    if args.mode == 'bm':
        if args.bmtype == 'matmul':
            if args.testname == 'np':
                compare = np_matmul
            elif args.testname == 'tc':
                compare = tc_matmul
            elif args.testname == 'tf':
                compare = tf_matmul
            else:
                print('unknown {} benchmark: {}'.format(args.bmtype, args.testcase))
                exit(1)
        elif args.bmtype == 'mlp':
            if args.testname == 'tc':
                compare = tc_mlp
            elif args.testname == 'tf':
                compare = tf_mlp
            else:
                print('unsupported {} benchmark: {}'.format(args.bmtype, args.testcase))
                exit(1)
        elif args.bmtype == 'mlp_grad':
            if args.testname == 'tc':
                compare = tc_mlp_grad
            elif args.testname == 'tf':
                compare = tf_mlp_grad
            else:
                print('unsupported {} benchmark: {}'.format(args.bmtype, args.testcase))
                exit(1)
        else:
            print('unknown benchmark: {}'.format(args.bmtype))
            exit(1)
        main(args.outdir, args.bmtype, args.testname, compare)
    elif args.mode == 'show':
        with open(args.filename) as f:
            results = json.load(f)
            plot_all(results.get('np', None), results.get('tc', None), results.get('tf', None))
    else:
        print('unknown mode {}'.format(args.mode))
        exit(1)
