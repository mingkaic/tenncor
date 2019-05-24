#!/usr/bin/env python3

''' Plugin interface definition '''

import abc

class PluginBase(metaclass=abc.ABCMeta):

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
