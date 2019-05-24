from gen.plugin_base import PluginBase

_plugin_id = "DTYPE"

@PluginBase.register
class DTypesPlugin:

    def plugin_id(self):
        return _plugin_id

    def process(self, generated_files, arguments):
        print('processing dtypes')
        print(arguments)
