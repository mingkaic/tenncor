import abc
import os.path

import tenncor as tc

_default_cachedir = '/tmp'

_sess_prefix = 'session_'
_sess_ext = '.onnx'

_env_prefix = 'env_'
_env_ext = '.bkup'

def _id_cachefile(fpath, prefix, ext):
    fname = os.path.basename(fpath)
    if os.path.isfile(fpath) and fname.startswith(prefix) and fname.endswith(ext):
        try:
            file_id = fname[len(prefix):len(fname) - len(ext)]
            return int(file_id, 16)
        except:
            print('ignoring invalid file {}'.format(fpath))
    return None

class CtxCache:
    @staticmethod
    def _format_cachefile(id):
        return _sess_prefix + hex(id)[2:] + _sess_ext

    def __init__(self, cache_dir=_default_cachedir):
        self.cache_dir = cache_dir
        dirs = os.listdir(self.cache_dir)
        self.cur_id = 0
        for el in dirs:
            fileid = _id_cachefile(os.path.join(self.cache_dir, el),
                _sess_prefix, _sess_ext)
            if fileid is not None:
                self.cur_id = max(self.cur_id, fileid)

    def backup(self, ctx):
        cache_fpath = os.path.join(self.cache_dir,
            CtxCache._format_cachefile(self.cur_id + 1))
        try:
            print('saving model "{}"'.format(cache_fpath))
            if tc.save_context_file(cache_fpath, ctx):
                print('successfully stored context to "{}"'.format(cache_fpath))
                self.cur_id += 1
                return True
        except Exception as e:
            print(e)
            print('failed storage to "{}"'.format(cache_fpath))
        return False

    def recover(self, ctx):
        cache_fpath = os.path.join(self.cache_dir,
            CtxCache._format_cachefile(self.cur_id))
        if os.path.isfile(cache_fpath):
            print('loading model from "{}"'.format(cache_fpath))
            roots = tc.load_context_file(cache_fpath, ctx)
            print('successfully recovered context from "{}"'.format(cache_fpath))
            return roots
        else:
            print('no such cache directory "{}"'.format(cache_fpath))
        return None

class EnvManager(metaclass=abc.ABCMeta):
    @staticmethod
    def _format_cachefile(id):
        return _env_prefix + hex(id)[2:] + _env_ext

    '''
    args:

    clean: whether to prevent auto-recovery on init.
    If clean is set to True, do not recover from existing cache, defaults to recovering

    '''
    def __init__(self, name,
        ctx=tc.global_context,
        default_init=None,
        clean=False,
        cacheroot=_default_cachedir):

        self.dirpath = os.path.join(cacheroot, name)
        if not os.path.isdir(self.dirpath):
            os.makedirs(self.dirpath)

        self.ctx_cache = CtxCache(self.dirpath)
        self.ctx = ctx

        # environment check
        dirs = os.listdir(self.dirpath)
        self.env_id = 0
        for el in dirs:
            fileid = _id_cachefile(os.path.join(self.dirpath, el),
                _env_prefix, _env_ext)
            if fileid is not None:
                self.env_id = max(self.env_id, fileid)

        try:
            if not clean:
                roots = self.ctx_cache.recover(self.ctx)
                if roots is not None and \
                    self._recover_env(os.path.join(self.dirpath,
                        EnvManager._format_cachefile(self.env_id))):
                    print('successful recovery')
                    print('recover session {}'.format(self.ctx_cache.cur_id))
                    print('recover environment {}'.format(self.env_id))
                    return
        except Exception as e:
            print('recovery error: {}'.format(e))

        if default_init is not None:
            default_init()

    def backup(self):
        self.ctx_cache.backup(self.ctx)
        if self._backup_env(os.path.join(self.dirpath,
            EnvManager._format_cachefile(self.env_id + 1))):
            self.env_id += 1

    @abc.abstractmethod
    def _backup_env(self, fpath: str) -> bool:
        '''
        Backup environment settings
        '''

    @abc.abstractmethod
    def _recover_env(self, fpath: str) -> bool:
        '''
        Recover environment settings
        '''
