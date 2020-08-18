
import tenncor as tc

import extenncor.trainer_cache as ecache
import extenncor.dataset_pb2 as ds_pb

import read_dataset as helper
import datasets_pb2

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
class OnxDSEnv(ecache.EnvManager):
    def __init__(self, name, oxfile, train_inputs,
                 connect_fn, trainstep_fn,
                 clean_startup=False, cachedir='/tmp',
                 optimize_cfg='', display_name = None,
                 ctx = tc.global_context):

        self.oxfile = oxfile
        def default_init():
            self.name = name
            self.dataset_idx = 0
            self.api = tc.TenncorAPI(self.ctx)

            # prevent shuffling to allow predictable recovery
            self.dataset = helper.load(self.oxfile)
            self.step = self.dataset.as_numpy_iterator()

            self.train_inputs = train_inputs
            labelled_inputs = [self.api.identity(train_input)
                for train_input in self.train_inputs]

            outputs = connect_fn(labelled_inputs, self.ctx)
            if not isinstance(outputs, Iterable):
                outputs = [outputs]
            self.train_outputs = [self.api.identity(train_output)
                for train_output in outputs]

            for i, linput in enumerate(labelled_inputs):
                linput.tag('recovery', 'input_{}'.format(i))

            for i, loutput in enumerate(self.train_outputs):
                loutput.tag('recovery', 'output_{}'.format(i))

            tc.optimize(optimize_cfg, self.ctx)

        if display_name is None:
            display_name = name
        super().__init__('tfds_' + display_name,
            ctx=ctx, default_init=default_init,
            clean=clean_startup, cacheroot=cachedir)
        self.trainstep_fn = trainstep_fn

    def _backup_env(self, fpath: str) -> bool:
        with open(fpath, 'wb') as envfile:
            oxds_env = ds_pb.OnxDSEnv()

            oxds_env.name = self.name
            oxds_env.dataset_idx = self.dataset_idx
            oxds_env.oxfile = self.oxfile

            envfile.write(oxds_env.SerializeToString())
            return True
        return False

    def _recover_env(self, fpath: str) -> bool:
        query = tc.Statement(self.ctx.get_actives())
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
            oxds_env = ds_pb.OnxDSEnv()
            oxds_env.ParseFromString(envfile.read())

            self.name = oxds_env.name
            self.dataset_idx = oxds_env.dataset_idx
            self.oxfile = oxds_env.oxfile

            print('recovered dataset {}'.format(self.oxfile))

            self.dataset = helper.load(self.oxfile)

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
            self.trainstep_fn(self.dataset_idx, self.ctx, data, self.train_inputs, self.train_outputs)
            self.dataset_idx += 1
            return True
        except StopIteration:
            return False
