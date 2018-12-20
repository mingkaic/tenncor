''' Reusable generator using plugin pattern '''

import os

import age.templates.template as template
import age.generator.internal as internal_plugin

def generate(fields, outpath = '', strip_prefix = '', plugins = [internal_plugin]):
    includepath = outpath
    if includepath and includepath.startswith(strip_prefix):
        includepath = includepath[len(strip_prefix):].strip("/")

    directory = {}
    for plugin in plugins:
        assert('process' in dir(plugin))
        directory = plugin.process(directory, includepath, fields)

    for akey in directory:
        afile = directory[akey]
        assert(isinstance(afile, template.AGE_FILE))
        if outpath:
            print(os.path.join(outpath, afile.fpath))
            with open(os.path.join(outpath, afile.fpath), 'w') as out:
                out.write(str(afile))
        else:
            print("============== %s ==============" % afile.fpath)
            print(str(afile))
