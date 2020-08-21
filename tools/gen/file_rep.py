#!/usr/bin/env python3

''' File representation '''

import os

class FileRep:

    def __init__(self, content,
        user_includes=[],
        internal_refs=[]):
        self.content = content
        self.includes = user_includes
        self.refs = internal_refs

    def generate(self, generated_path):
        includes = self.includes + [
            '"' + os.path.join(generated_path, ref) + '"'
            for ref in self.refs
        ]

        return '\n'.join(['#include {}'.format(include)
            for include in includes]) + '\n\n' + str(self.content)
