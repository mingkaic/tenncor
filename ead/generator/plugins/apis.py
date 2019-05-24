from gen.plugin_base import PluginBase

_plugin_id = "API"

@PluginBase.register
class APIsPlugin:

    def plugin_id(self):
        return _plugin_id

    def process(self, generated_files, arguments):
        print('processing apis')
        print(arguments)
