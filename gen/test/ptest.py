#!/usr/bin/env python3

import io
import logging
import unittest

from gen.generate import generate
from gen.file_rep import FileRep
from gen.plugin_base import PluginBase
from gen.dump import GenDumpBase

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

class ClientTest(unittest.TestCase):
    def test_nonregistered_plugin(self):
        with self.assertLogs(level=logging.WARNING) as cm:
            generate({'abc': 123}, plugins = ['abc'])
            self.assertEqual(cm.output, ['WARNING:root:'+\
                'plugin abc is not a registered plugin base:skipping plugin'])

    def test_empty_plugin(self):
        ''' This mock plugin returns no generated files '''
        @PluginBase.register
        class EmptyPlugin:

            def plugin_id(self):
                return 'empty'

            def process(self, generated_files, arguments):
                return

        with self.assertLogs(level=logging.WARNING) as cm:
            generate({'abc': 123}, plugins = [EmptyPlugin()])
            self.assertEqual(cm.output, ['WARNING:root:plugin empty failed '+\
                'to return dictionary of files:ignoring plugin output'])

    def test_faulty_plugin(self):
        ''' This mock plugin returns a dictionary of non-FileReps '''
        @PluginBase.register
        class FaultyPlugin:

            def plugin_id(self):
                return 'faulty'

            def process(self, generated_files, arguments):
                return {
                    'mock_faulty': {'a': 1, '3': 'c', '@': 2}
                }

        with self.assertLogs(level=logging.WARNING) as cm:
            generate({'abc': 123}, plugins = [FaultyPlugin()])
            self.assertEqual(cm.output, ['WARNING:root:generated representation '+\
                '{\'a\': 1, \'3\': \'c\', \'@\': 2} '+\
                'is not a FileRep:skipping file'])

    def test_generate_plugin(self):
        @PluginBase.register
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
                return {
                    self.outfile: FileRep(content,
                        user_includes=self.includes,
                        internal_refs=refs),
                    **generated_files
                }

        @GenDumpBase.register
        class MockDump:

            def __init__(self):
                self.files = {}

            def write(self, filename, file):
                out_content = file.generate('')
                self.files[filename] = out_content

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
    unittest.main()
