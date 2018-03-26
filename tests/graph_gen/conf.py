#!/usr/bin/env python3

import yaml
import os

def getOpts(yConf):
	dir_path = os.path.dirname(os.path.realpath(__file__))
	conf_path = os.path.join(dir_path, yConf)

	stryaml = open(conf_path, "r")
	res = yaml.load(stryaml)
	stryaml.close()
	return res
