#!/usr/bin/env python3

''' Plugin interface definition '''

import abc

class PluginBase(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def plugin_id(self) -> str:
        '''
        Return plugin identifier
        '''

    @abc.abstractmethod
    def process(self,
        generated_files: dict, arguments: dict, **kwargs) -> dict:
        '''
        Given
            output path
            dictionary of generated_files (
                mapping filename to FileRep) and
            dictionary of arguments,
        Return {filename: FileRep}
        '''
