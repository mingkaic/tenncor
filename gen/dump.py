#usr/bin/env python3

''' Dump interface for collecting generate output files '''

import abc
import logging
import os

from gen.file_rep import FileRep

class GenDumpBase(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def write(self, filename: str, file: FileRep):
        '''
        Given
            filename of generated file
            file representation
        Write/Store generated file
        '''

@GenDumpBase.register
class PrintDump:

    def write(self, filename, file):
        out_content = file.generate('')
        print('============== {} =============='.format(filename))
        print(out_content)

@GenDumpBase.register
class FileDump:

    '''
    outpath specifies the file path to write to
    includepath specifies the include path
        used by the file when generating content
    '''
    def __init__(self, outpath, includepath=None):
        self.outpath = outpath
        if includepath is None:
            self.includepath = outpath
        else:
            self.includepath = includepath

    def write(self, filename, file):
        out_content = file.generate(self.includepath)
        filepath = os.path.join(self.outpath, filename)
        logging.info('generating %s', filepath)
        with open(filepath, 'w') as outf:
            outf.write(out_content)
