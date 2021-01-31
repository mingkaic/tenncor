import sys
import json
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

def main(bmtype, testname, measure):
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
    with open('{}_results.json'.format(bmtype), 'w') as f:
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
    mode = sys.argv[1]
    bmtype = sys.argv[2]
    if len(sys.argv) > 3:
        testname = sys.argv[3]
    if mode == 'bm':
        if bmtype == 'matmul':
            if testname == 'np':
                compare = np_matmul
            elif testname == 'tc':
                compare = tc_matmul
            elif testname == 'tf':
                compare = tf_matmul
            else:
                print('unknown {} benchmark: {}'.format(bmtype, testcase))
                exit(1)
        elif bmtype == 'mlp':
            if testname == 'tc':
                compare = tc_mlp
            elif testname == 'tf':
                compare = tf_mlp
            else:
                print('unsupported {} benchmark: {}'.format(bmtype, testcase))
                exit(1)
        elif bmtype == 'mlp_grad':
            if testname == 'tc':
                compare = tc_mlp_grad
            elif testname == 'tf':
                compare = tf_mlp_grad
            else:
                print('unsupported {} benchmark: {}'.format(bmtype, testcase))
                exit(1)
        else:
            print('unknown benchmark: {}'.format(bmtype))
            exit(1)
        main(bmtype, testname, compare)
    elif mode == 'show':
        with open(bmtype) as f:
            results = json.load(f)
            plot_all(results.get('np', None), results.get('tc', None), results.get('tf', None))
    else:
        print('unknown mode {}'.format(mode))
        exit(1)
