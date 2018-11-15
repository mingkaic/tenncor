''' Represent files as templates and recursively format templates according to some dictionary of values '''

class FILE_REPR:
	def __init__(self, template):
		self.template = template

	def repr(self, values):
		items = self.__dict__.items()
		fmt = {}
		for key, value in items:
			if key is 'template':
				continue
			arg, func = value
			if arg in values:
				fmt[key] = func(values[arg])
		return self.template.format(**fmt)
