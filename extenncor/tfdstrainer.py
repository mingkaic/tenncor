
import tenncor as tc

import extenncor.trainer_cache as ecache
import extenncor.dataset_pb2 as ds_pb

import tensorflow_datasets as tfds
from collections.abc import Iterable

_tfds_nio_limit = 100
_tfds_args = {
    'split': None,
    'data_dir': None,
    'batch_size': None,
    'in_memory': None,
    'download': True,
    'as_supervised': False,
    'try_gcs': False,
}

@ecache.EnvManager.register
class TfdsEnv(ecache.EnvManager):
    def __init__(self, name, sess, train_inputs,
                 connect_fn, trainstep_fn,
                 clean_startup=False, cachedir='/tmp',
                 optimize_cfg='', display_name = None,
                 **kwargs):
        def default_init():
            self.name = name
            ds_params = dict(kwargs)
            self.ds_params = dict([(key, ds_params[key])
                for key in ds_params
                if key in _tfds_args])
            self.dataset_idx = 0

            # prevent shuffling to allow predictable recovery
            self.dataset = tfds.load(name, shuffle_files=False, **self.ds_params)
            self.step = self.dataset.as_numpy_iterator()

            self.train_inputs = train_inputs
            labelled_inputs = [tc.identity(train_input) for train_input in self.train_inputs]

            outputs = connect_fn(labelled_inputs)
            if not isinstance(outputs, Iterable):
                outputs = [outputs]
            self.train_outputs = [tc.identity(train_output) for train_output in outputs]

            for i, linput in enumerate(labelled_inputs):
                linput.tag('recovery', 'input_{}'.format(i))

            for i, loutput in enumerate(self.train_outputs):
                loutput.tag('recovery', 'output_{}'.format(i))

            self.sess.track(self.train_outputs)
            tc.optimize(self.sess, optimize_cfg)

        if display_name is None:
            display_name = name
        super().__init__('tfds_' + display_name, sess,
                         default_init=default_init,
                         clean=clean_startup,
                         cacheroot=cachedir)
        self.trainstep_fn = trainstep_fn

    def _backup_env(self, fpath: str) -> bool:
        with open(fpath, 'wb') as envfile:
            tfds_env = ds_pb.TfdsEnv()

            tfds_env.name = self.name
            tfds_env.dataset_idx = self.dataset_idx

            for arg in _tfds_args:
                ds_val = self.ds_params.get(arg, _tfds_args[arg])
                if ds_val is not None:
                    setattr(tfds_env.ds_params, arg, ds_val)
                else:
                    setattr(tfds_env.ds_params, arg + '_nil', True)

            envfile.write(tfds_env.SerializeToString())
            return True
        return False

    def _recover_env(self, fpath: str) -> bool:
        query = tc.Statement(self.sess.get_tracked())
        self.train_inputs = []
        self.train_outputs = []

        for i in range(_tfds_nio_limit):
            detections = query.find('''{
                "op":{
                    "opname":"IDENTITY",
                    "attrs":{
                        "recovery":{
                            "str":"input_%d"
                        }
                    },
                    "args":{
                        "symb":"recovery_target"
                    }
                }
            }''' % i, sym_cap='recovery_target')
            if len(detections) > 0:
                self.train_inputs.append(tc.to_variable(detections[0]))
            else:
                break

        for i in range(_tfds_nio_limit):
            detections = query.find('''{
                "op":{
                    "opname":"IDENTITY",
                    "attrs":{
                        "recovery":{
                            "str":"output_%d"
                        }
                    }
                }
            }''' % i)
            if len(detections) > 0:
                self.train_outputs.append(detections[0])
            else:
                break

        print('loading environment from "{}"'.format(fpath))
        with open(fpath, 'rb') as envfile:
            tfds_env = ds_pb.TfdsEnv()
            tfds_env.ParseFromString(envfile.read())

            self.name = tfds_env.name
            self.dataset_idx = tfds_env.dataset_idx

            self.ds_params = dict([(arg, getattr(tfds_env.ds_params, arg))
                                   for arg in _tfds_args
                                   if not getattr(tfds_env.ds_params, arg + '_nil')])
            print('recovered with dataset params: {}'.format(self.ds_params))

            self.dataset = tfds.load(self.name, shuffle_files=False, **self.ds_params)

            self.step = self.dataset.as_numpy_iterator()
            print('skipping the first {} images'.format(self.dataset_idx))
            for _ in range(self.dataset_idx):
                next(self.step)

            print('successfully recovered environment from "{}"'.format(fpath))
            return True

        print('failed environment recovery')
        return False

    def train(self):
        try:
            data = next(self.step)
            self.trainstep_fn(self.dataset_idx, self.sess, data, self.train_inputs, self.train_outputs)
            self.dataset_idx += 1
            return True
        except StopIteration:
            return False
