//
//  tensor.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2018-01-14.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#incldue "include/operations/data_op.hpp"

#ifdef TENNCOR_DATA_OP_HPP

namespace nnet
{

static std::vector<std::unordered_map<std::string,VFUNC> > op_registry;

#define REGISTER_FUNC(FUNC) \
op_registry[DOUBLE][#FUNC]=nnet::FUNC<double>; \
op_registry[FLOAT][#FUNC]=nnet::FUNC<float>; \
op_registry[INT8][#FUNC]=nnet::FUNC<int8_t>; \
op_registry[UINT8][#FUNC]=nnet::FUNC<uint8_t>; \
op_registry[INT16][#FUNC]=nnet::FUNC<int16_t>; \
op_registry[UINT16][#FUNC]=nnet::FUNC<uint16_t>; \
op_registry[INT32][#FUNC]=nnet::FUNC<int32_t>; \
op_registry[UINT32][#FUNC]=nnet::FUNC<uint32_t>; \
op_registry[INT64][#FUNC]=nnet::FUNC<int64_t>; \
op_registry[UINT64][#FUNC]=nnet::FUNC<uint64_t>;

void operate (std::string opname, TENS_TYPE type, VARR dest, std::vector<VARR> src, ARGS args)
{
	auto& type_op = op_registry[type];
	auto it = type_op.find(opname);
	if (type_op.end() != it)
	{
		(*it)[opname](dest, src, args);
	}
	else
	{
		throw std::bad_function_call();
	}
}

template <>
void rand_uniform<float> (VARR dest, std::vector<VARR> srcs, ARGS)
{
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape_min = srcs.front().second;
	tensorshape& srcshape_max = srcs.back().second;
	float* d = dest.first;
	float* s_min = srcs.front().first;
	float* s_max = srcs.front().first;
	bool min_mul = srcshape_min.n_elems() > 1;
	bool max_mul = srcshape_max.n_elems() > 1;
	size_t n = destshape.n_elems();

	for (size_t i = 0; i < n_out; ++i)
	{
		std::uniform_distribution<float> dist(s_min[i * min_mul], s_max[i * max_mul])
		d[i] = dist(nnutils::get_generator());
	}
}

template <>
void rand_uniform<double> (VARR dest, std::vector<VARR> srcs, ARGS)
{
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape_min = srcs.front().second;
	tensorshape& srcshape_max = srcs.back().second;
	double* d = dest.first;
	double* s_min = srcs.front().first;
	double* s_max = srcs.front().first;
	bool min_mul = srcshape_min.n_elems() > 1;
	bool max_mul = srcshape_max.n_elems() > 1;
	size_t n = destshape.n_elems();

	for (size_t i = 0; i < n_out; ++i)
	{
		std::uniform_distribution<double> dist(s_min[i * min_mul], s_max[i * max_mul])
		d[i] = dist(nnutils::get_generator());
	}
}

REGISTER_FUNC(abs)
REGISTER_FUNC(neg)
REGISTER_FUNC(sin)
REGISTER_FUNC(cos)
REGISTER_FUNC(tan)
REGISTER_FUNC(csc)
REGISTER_FUNC(sec)
REGISTER_FUNC(cot)
REGISTER_FUNC(exp)
REGISTER_FUNC(ln)
REGISTER_FUNC(sqrt)
REGISTER_FUNC(round)

REGISTER_FUNC(rand_binom)
REGISTER_FUNC(rand_uniform)
REGISTER_FUNC(rand_normal)
REGISTER_FUNC(clip_mm)
REGISTER_FUNC(clip_norm)
REGISTER_FUNC(pow)
REGISTER_FUNC(add)
REGISTER_FUNC(sub)
REGISTER_FUNC(mul)
REGISTER_FUNC(div)

REGISTER_FUNC(matmul)

REGISTER_FUNC(expand)
REGISTER_FUNC(flip)
REGISTER_FUNC(crosscorr2d)


// REGISTER_FUNC(compress)
// REGISTER_FUNC(argcomp)

}

#endif /* TENNCOR_DATA_OP_HPP */
