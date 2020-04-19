import unittest
import os.path
import tempfile

import tenncor as tc
from dbg.print import graph_to_str

class LAYRTest(unittest.TestCase):
    def test_session_save(self):
        sess = tc.Session()

        nunits = 9
        ninput = 10
        noutput = int(ninput / 2)
        n_batch = 10

        model = tc.layer.link([
            tc.layer.dense([ninput], [nunits],
                weight_init=tc.unif_xavier_init(),
                bias_init=tc.zero_init()),
            tc.layer.bind(tc.sigmoid),
            tc.layer.dense([nunits], [noutput],
                weight_init=tc.unif_xavier_init(),
                bias_init=tc.zero_init()),
            tc.layer.bind(tc.sigmoid),
        ])
        train_input = tc.EVariable([n_batch, ninput])
        train_output = tc.EVariable([n_batch, noutput])

        train_err = tc.apply_update([model],
            lambda err, vars: tc.approx.sgd(err, vars, learning_rate=0.9),
            lambda models: tc.error.sqr_diff(train_output, models[0].connect(train_input)))
        sess.track([train_err])

        temp_dir = tempfile.mkdtemp()
        test_file = os.path.join(temp_dir, 'layr_test')
        tc.save_session_file(test_file, sess)

        sess2 = tc.Session()
        tc.load_session_file(test_file, sess2)
        roots = sess2.get_tracked()
        self.assertEquals(1, len(roots))
        self.assertEquals(graph_to_str(train_err),
            graph_to_str(list(roots)[0]))

    def test_gru(self):
        brain = tc.layer.link([
            tc.layer.conv([5, 5], 1, 16,
                weight_init=tc.norm_xavier_init(0.5),
                zero_padding=[2, 2]), # input of [1, 2641, 128, 1] -> output of [16, 2647, 128, 1]
            tc.layer.conv([5, 5], 16, 20,
                weight_init=tc.norm_xavier_init(0.5),
                zero_padding=[2, 2]), # input of [16, 2647, 128, 1] -> output of [20, 2647, 128, 1]
            tc.layer.gru(tc.Shape([2647, 20]), 20, 128,
                seq_dim = 2,
                weight_init=tc.zero_init(),
                bias_init=tc.zero_init()), # input of [20, 2647, 128, 1] -> output of [20, 2647, 128]
        ], tc.EVariable([128, 2647, 1], label='input'))

        # state = [x, 2647]
        # concat([20, 2647] [x, 2647]) -> [20 + x, 2647]
        # inshape = [20 + x, 2647]
        # [20 + x, 2647] @ [x, 20 + x] -> [x, 2647]
        self.assertEquals([128, 2647, 20], list(brain.shape()))


if __name__ == "__main__":
    unittest.main()
