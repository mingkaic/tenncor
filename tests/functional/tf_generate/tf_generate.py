#!/usr/bin/env python3
import sys
import os
import tensorflow as tf
import tf_sampler as sam
import numpy as np
import random

def clip(t):
	r = np.random.rand(2)
	minr = sam.register(min(r[0], r[1]))
	maxr = sam.register(max(r[0], r[1]))
	return tf.clip_by_value(t, minr, maxr)

UNAR = {
	"abs": tf.abs,
	"neg": tf.neg,
	"sin": tf.sin,
	"cos": tf.cos,
	"tan": tf.tan,
	"exp": tf.exp,
	"sqrt": tf.sqrt,
	"log": tf.log,
}

UNAR_SCALR = {
	"clip": clip,
	"clip_norm": (lambda t: tf.clip_by_norm(t, sam.register(random.random() * 10 + 0.0001))),
	"pow": (lambda t: tf.pow(t,sam.register(random.random() * 6 - 3))),
	"add_c": (lambda t: tf.add(t, sam.register(random.random()))),
	"sub_c": (lambda t: tf.sub(t, sam.register(random.random()))),
	"c_sub": (lambda t: tf.sub(sam.register(random.random()), t)),
	"mul_c": (lambda t: tf.mul(t, sam.register(random.random()))),
	"div_c": (lambda t: tf.div(t, sam.register(random.random()))),
	"c_div": (lambda t: tf.div(sam.register(random.random()), t)),
}

BINAR = {
	"add": (lambda t1, t2: tf.add(t1, t2)),
	"sub": (lambda t1, t2: tf.sub(t1, t2)),
	"mul": (lambda t1, t2: tf.mul(t1, t2)),
	"div": (lambda t1, t2: tf.div(t1, t2)),
	# "equal": (lambda t1, t2: tf.equal(t1, t2)),
	# "not_equal": (lambda t1, t2: tf.not_equal(t1, t2)),
}

TRANS = {
	"transpose": tf.transpose,
	"reduce_max_i": (lambda t: tf.reduce_max(t, sam.register(random.randint(0, t.get_shape().ndims-1)))),
	"reduce_min_i": (lambda t: tf.reduce_min(t, sam.register(random.randint(0, t.get_shape().ndims-1)))),
	"reduce_sum_i": (lambda t: tf.reduce_sum(t, sam.register(random.randint(0, t.get_shape().ndims-1)))),
	"reduce_mean_i": (lambda t: tf.reduce_mean(t, sam.register(random.randint(0, t.get_shape().ndims-1)))),
}

dir = "samples"
def run(outfile, opname):
	genOp = sam.unar_op
	with open(os.path.join(dir, outfile + ".csv"), 'a') as f:
		if outfile == "UNAR":
			opmap = UNAR
		elif outfile == "UNAR_SCALR":
			opmap = UNAR_SCALR
		elif outfile == "TRANS":
			opmap = TRANS
		elif outfile == "BINAR":
			genOp = sam.bin_elem_op
			opmap = BINAR
		
		if outfile != "MATMUL":
			if (opname == "transpose"):
				shape = sam.rand_shape(2)
			else:
				shape = sam.rand_shape()
			f.write(genOp(opname, opmap[opname], shape) + '\n')
		else:
			sh = sam.rand_shape(2)
			f.write(sam.bin_trans_op("matmul" + opname, tf.matmul, sh, [sh[1], random.randint(33, 100)]) + '\n')
	sam.clearScalr()

def single_run():
	outfile = "MATMUL"
	opname = ""
	if len(sys.argv) > 1:
		outfile = sys.argv[1]
	if len(sys.argv) > 2:
		opname = sys.argv[2]
	run(outfile, opname)

def multiple_run():
	for opname in UNAR:
		run("UNAR", opname)

	for opname in UNAR_SCALR:
		run("UNAR_SCALR", opname)

	for opname in BINAR:
		run("BINAR", opname)

	for opname in TRANS:
		run("TRANS", opname)

	for i in range(10):
		run("MATMUL", str(i))

if __name__ == "__main__":
	if os.environ.get('SINGLE_RUN'):
		single_run()
	else:
		multiple_run()
