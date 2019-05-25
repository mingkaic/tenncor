#!/usr/bin/env python3

''' Reusable generator using plugin pattern '''

import io
import os
import logging

try: # this is a workaround (todo: remove)
    from gen.plugin_base import PluginBase
    from gen.dump import GenDumpBase, PrintDump
except:
    from gen.plugin_base2 import PluginBase
    from gen.dump2 import GenDumpBase, PrintDump

from gen.file_rep import FileRep

def generate(fields, out = PrintDump(), plugins = []):
    if not isinstance(out, GenDumpBase):
        logging.warning("unknown output dump %s: will not generate", out)
        return

    generated_files = dict()
    for plugin in plugins:
        if not isinstance(plugin, PluginBase):
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
        out.write(filename, file_rep)
