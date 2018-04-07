''' AST graph traversal utility '''

import graphast.nodes as nodes

# bottom up traverse tree
def traverse(root, func):
	collection = []
	deps = []
	if isinstance(root, nodes.node):
		for arg in root.args:
			depid, coll = traverse(arg, func)
			collection.extend(coll)
			deps.append(depid)
	id, info = func(root, deps)
	collection.append(info)
	return id, collection
