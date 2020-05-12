import yaml

_template_prefixes = ['typename', 'class']

def strip_template_prefix(template):
    for template_prefix in _template_prefixes:
        if template.strip().startswith(template_prefix):
            return template[len(template_prefix):].strip()
    # todo: parse valued templates variable (e.g.: size_t N)
    return template

def reference_classes(classes):
    out = []
    for clas in classes:
        if isinstance(clas, str):
            with open(clas) as f:
                obj = yaml.safe_load(f.read())
                if type(obj) != dict:
                    raise Exception('cannot treat non-dictionary object as class: {}'.format(clas))
                out.append(obj)
        else:
            out.append(clas)
    return out

def order_classes(classes):
    cnames = []
    edges = dict()
    for clas in classes:
        reqs = clas.get('requires', [])
        if not isinstance(reqs, list):
            reqs = [reqs]
        cname = clas['name']
        cnames.append(cname)
        for req in reqs:
            if req not in edges:
                edges[req] = []
            edges[req].append(cname)

    # use reverse bellman-ford to calculate distance
    nvertices = len(cnames)
    # maximum distance from source to dest
    # where dest is any class that is not required by any other
    distance = dict([(clas, 0) for clas in cnames])

    dests = list(filter(lambda v: len(edges.get(v, [])) == 0, cnames))
    if len(dests) == 0:
        raise Exception('Circular dependency: no independent class')
    for dest in dests:
        distance[dest] = 0
    for _ in range(1,nvertices):
        for src in edges:
            for dest in edges[src]:
                if distance[src] < distance[dest] + 1:
                    distance[src] = distance[dest] + 1
    for src in edges:
        for dest in edges[src]:
            if distance[src] < distance[dest] + 1:
                raise Exception("Circular dependency: circ dependence between {} and {}".format(src, dst))
    # sources are classes that are not required by anything else
    # we want to first declare classes that are required, so reverse sort classes by distance
    cnames.sort(key=lambda c: distance[c], reverse=True)
    return cnames
