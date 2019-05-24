#!/usr/bin/env python2

''' Plugin interface definition '''

import abc

# this is a workaround because bazel python interpeter settings duck sick
class PluginBase:
    __metaclass__=abc.ABCMeta

    def get_iterator(self):
        return self.__iter__()

    @abc.abstractmethod
    def plugin_id(self):
        """
        Return plugin identifier
        """

    @abc.abstractmethod
    def process(self, generated_files, arguments):
        """
        Given
            output path
            dictionary of generated_files (
                mapping filename to FileRep) and
            dictionary of arguments,
        Return {filename: FileRep}
        """
