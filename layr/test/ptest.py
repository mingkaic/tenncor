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

        model = tc.link([
            (tc.dense_name, tc.layer.dense([ninput], [nunits],
                weight_init=tc.unif_xavier_init(),
                bias_init=tc.zero_init())),
            (tc.bind_name, tc.bind(tc.sigmoid)),
            (tc.dense_name, tc.layer.dense([nunits], [noutput],
                weight_init=tc.unif_xavier_init(),
                bias_init=tc.zero_init())),
            (tc.bind_name, tc.bind(tc.sigmoid)),
        ])
        train_input = tc.EVariable([n_batch, ninput])
        train_output = tc.EVariable([n_batch, noutput])
        train_err = tc.sgd_train(tc.link_name, model, train_input, train_output,
            lambda assocs: tc.approx.sgd(assocs, 0.9))
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


if __name__ == "__main__":
    unittest.main()
