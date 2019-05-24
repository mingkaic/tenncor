#!/usr/bin/env python3

''' Reusable generator using plugin pattern '''

import io
import os
import logging

try: # this is a workaround (todo: remove)
    from gen.plugin_base import PluginBase
except:
    from gen.plugin_base2 import PluginBase

from gen.file_rep import FileRep

def generate(fields, outpath = None, plugins = []):
    generated_files = dict()
    for plugin in plugins:
        if not issubclass(type(plugin), PluginBase):
            logging.warning('plugin %s is not a registered plugin base:'+\
                'skipping plugin', plugin)
            continue
        plugin_id = plugin.plugin_id()
        logging.info('processing plugin %s', plugin_id)
        plugin_files = plugin.process(generated_files, fields)
        if type(plugin_files) is not dict:
            logging.warning('plugin %s failed to return dictionary of files:'+\
                'ignoring plugin output', plugin_id)
        else:
            generated_files = plugin_files

    for filename in generated_files:
        file_rep = generated_files[filename]
        if not isinstance(file_rep, FileRep):
            logging.warning('generated representation %s is not a FileRep:'+\
                'skipping file', file_rep)
            continue
        if type(outpath) is str and len(outpath) > 0:
            out_content = file_rep.generate(outpath)
            filepath = os.path.join(outpath, filename)
            logging.info('generating %s', filepath)
            with open(filepath, 'w') as out:
                out.write(out_content)
        else:
            out_content = file_rep.generate('')
            file_out = '============== {} ==============\n'.format(
                filename)+out_content+'\n'
            if issubclass(type(outpath), io.IOBase) and outpath.writable():
                outpath.write(file_out)
            else:
                print(file_out)
