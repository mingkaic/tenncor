#!/usr/bin/env python3

import io
import logging
import unittest

from gen.generate import generate
from gen.file_rep import FileRep
from gen.plugin_base2 import PluginBase
from gen.dump2 import GenDumpBase

_test_generate_plugin_a = '''
#include "ma_name.hpp"
#include "ya_name.h"
#include <abc>

abcde12345mock_content
'''.strip()

_test_generate_plugin_b = '''
#include "his_name.hpp"
#include "her_name.h"
#include <xyz>
#include "mock_a"

fghij67890mock_content
'''.strip()

class MockLoggingHandler(logging.Handler):
    def __init__(self, *args, **kwargs):
        self.reset()
        logging.Handler.__init__(self, *args, **kwargs)

    def emit(self, record):
        self.messages[record.levelname.lower()].append(record.getMessage())

    def reset(self):
        self.messages = {
            'debug': [],
            'info': [],
            'warning': [],
            'error': [],
            'critical': [],
        }

_log_handler = MockLoggingHandler()

class ClientTest(unittest.TestCase):
    def setUp(self):
        _log_handler.reset()

    def test_nonregistered_plugin(self):
        generate({'abc': 123}, plugins = ['abc'])
        messages = _log_handler.messages['warning']
        self.assertEqual(messages, [
            'plugin abc is not a registered plugin base:skipping plugin'])

    def test_empty_plugin(self):
        ''' This mock plugin returns no generated files '''
        class EmptyPlugin:

            def plugin_id(self):
                return 'empty'

            def process(self, generated_files, arguments):
                return

        PluginBase.register(EmptyPlugin)

        generate({'abc': 123}, plugins = [EmptyPlugin()])
        messages = _log_handler.messages['warning']
        self.assertEqual(messages, ['plugin empty failed '+\
            'to return dictionary of files:ignoring plugin output'])

    def test_faulty_plugin(self):
        ''' This mock plugin returns a dictionary of non-FileReps '''
        class FaultyPlugin:

            def plugin_id(self):
                return 'faulty'

            def process(self, generated_files, arguments):
                return {
                    'mock_faulty': {'a': 1, '3': 'c', '@': 2}
                }

        PluginBase.register(FaultyPlugin)

        generate({'abc': 123}, plugins = [FaultyPlugin()])
        messages = _log_handler.messages['warning']
        self.assertEqual(messages, ['generated representation '+\
            '{\'a\': 1, \'@\': 2, \'3\': \'c\'} '+\
            'is not a FileRep:skipping file'])

    def test_generate_plugin(self):
        class Plugin:
            def __init__(self, outfile, content, includes):
                self.outfile = outfile
                self.content = content
                self.includes = includes

            def plugin_id(self):
                return 'plugin'

            def process(self, generated_files, arguments):
                # assert generated_files entered init file
                if len(generated_files) > 0:
                    refs = [list(generated_files.keys())[0]]
                else:
                    refs = []
                content = self.content + str(arguments)
                return generated_files.update({
                    self.outfile: FileRep(content,
                        user_includes=self.includes,
                        internal_refs=refs)
                })

        PluginBase.register(Plugin)

        class MockDump:

            def __init__(self):
                self.files = {}

            def write(self, filename, file):
                out_content = file.generate('')
                self.files[filename] = out_content

        GenDumpBase.register(MockDump)

        a_file = 'mock_a'
        a_content = 'abcde12345'
        a_includes = ['"ma_name.hpp"', '"ya_name.h"', '<abc>']

        b_file = 'mock_b'
        b_content = 'fghij67890'
        b_includes = ['"his_name.hpp"', '"her_name.h"', '<xyz>']

        plugins = [
            Plugin(a_file, a_content, a_includes),
            Plugin(b_file, b_content, b_includes),
        ]
        test_out = MockDump()
        generate('mock_content', out=test_out, plugins = plugins)

        dump = test_out.files
        self.assertTrue(a_file in dump,
            '{} not found in {}'.format(a_file, dump))
        self.assertTrue(b_file in dump,
            '{} not found in {}'.format(b_file, dump))

        self.assertEqual(_test_generate_plugin_a,
            dump[a_file].strip())
        self.assertEqual(_test_generate_plugin_b,
            dump[b_file].strip())

if __name__ == "__main__":

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(_log_handler)

    unittest.main()
