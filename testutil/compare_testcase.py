import json
import os.path

def _npconvert(val):
    if 'tolist' in dir(val):
        return val.tolist()
    return val

class TestWriter:
    def __init__(self, outfile):
        self.outfile = outfile
        if os.path.isfile(outfile):
            try:
                with open(outfile, 'r') as file:
                    self.cases = json.load(file)
                return
            except:
                pass
        self.cases = {}

    def write(self):
        with open(self.outfile, 'w') as file:
            file.write(json.dumps(self.cases))

    def save_case(self, case_name, ins, out):
        ins = [_npconvert(inp) for inp in ins]
        out = _npconvert(out)
        if case_name not in self.cases:
            self.cases[case_name] = []
        self.cases[case_name].append({
            'input': ins,
            'output': out
        })

class TestReader:
    def __init__(self, infile):
        with open(infile, 'r') as file:
            self.cases = json.load(file)

    def get_case(self, case_name):
        cases = self.cases[case_name]
        return [(case.get('input', ()), case.get('output', None)) for case in cases]
