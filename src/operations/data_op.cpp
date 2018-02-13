//
//  data_op.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2018-01-14.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/operations/data_op.hpp"

#ifdef TENNCOR_DATA_OP_HPP

namespace nnet
{

using VFUNC_MAP = std::unordered_map<TENS_TYPE,VFUNC>;

using FUNC_REG = std::unordered_map<std::string,VFUNC_MAP>;

#define REGISTER_FUNC(FUNC) {#FUNC, VFUNC_MAP{\
{DOUBLE, nnet::FUNC<double>},{FLOAT, nnet::FUNC<float>},\
{INT8, nnet::FUNC<int8_t>},{UINT8, nnet::FUNC<uint8_t>},\
{INT16, nnet::FUNC<int16_t>},{UINT16, nnet::FUNC<uint16_t>},\
{INT32, nnet::FUNC<int32_t>},{UINT32, nnet::FUNC<uint32_t>},\
{INT64, nnet::FUNC<int64_t>},{UINT64, nnet::FUNC<uint64_t>}}},

static const FUNC_REG op_registry = {
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
	REGISTER_FUNC(argmax)
	REGISTER_FUNC(max)
	REGISTER_FUNC(sum)

	REGISTER_FUNC(pow)
	REGISTER_FUNC(add)
	REGISTER_FUNC(sub)
	REGISTER_FUNC(mul)
	REGISTER_FUNC(div)
	REGISTER_FUNC(eq)
	REGISTER_FUNC(neq)
	REGISTER_FUNC(lt)
	REGISTER_FUNC(gt)
	REGISTER_FUNC(rand_binom)
	REGISTER_FUNC(rand_uniform)
	REGISTER_FUNC(rand_normal)
};

std::unordered_set<std::string> all_ops (void)
{
	std::unordered_set<std::string> opset;
	for (auto& op_pair : op_registry)
	{
		opset.emplace(op_pair.first);
	}
	return opset;
}

void operate (std::string opname, TENS_TYPE type, VARR dest, std::vector<VARR> src)
{
	auto type_it = op_registry.find(opname);
	if (op_registry.end() != type_it)
	{
		auto& type_map = type_it->second;
		auto it = type_map.find(type);
		if (type_map.end() != it)
		{
			(it->second)(dest, src);
			return;
		}
	}
	throw std::bad_function_call();
}

template <>
void abs<uint8_t> (VARR dest, std::vector<VARR> srcs)
{
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcshape);
	size_t n = dest.second.n_elems();
	std::memcpy(dest.first, srcs.front().first, sizeof(uint8_t) * n);
}

template <>
void abs<uint16_t> (VARR dest, std::vector<VARR> srcs)
{
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcshape);
	size_t n = dest.second.n_elems();
	std::memcpy(dest.first, srcs.front().first, sizeof(uint16_t) * n);
}

template <>
void abs<uint32_t> (VARR dest, std::vector<VARR> srcs)
{
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcshape);
	size_t n = dest.second.n_elems();
	std::memcpy(dest.first, srcs.front().first, sizeof(uint32_t) * n);
}

template <>
void abs<uint64_t> (VARR dest, std::vector<VARR> srcs)
{
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcshape);
	size_t n = dest.second.n_elems();
	std::memcpy(dest.first, srcs.front().first, sizeof(uint64_t) * n);
}

template <>
void neg<uint8_t> (VARR, std::vector<VARR>) { throw std::bad_function_call(); }

template <>
void neg<uint16_t> (VARR, std::vector<VARR>) { throw std::bad_function_call(); }

template <>
void neg<uint32_t> (VARR, std::vector<VARR>) { throw std::bad_function_call(); }

template <>
void neg<uint64_t> (VARR, std::vector<VARR>) { throw std::bad_function_call(); }

template <>
void rand_uniform<float> (VARR dest, std::vector<VARR> srcs)
{
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape_min = srcs.front().second;
	tensorshape& srcshape_max = srcs.back().second;
	float* d = (float*) dest.first;
	float* s_min = (float*) srcs.front().first;
	float* s_max = (float*) srcs.front().first;
	bool min_mul = srcshape_min.n_elems() > 1;
	bool max_mul = srcshape_max.n_elems() > 1;
	size_t n = destshape.n_elems();

	for (size_t i = 0; i < n; ++i)
	{
		std::uniform_real_distribution<float> dist(s_min[i * min_mul], s_max[i * max_mul]);
		d[i] = dist(nnutils::get_generator());
	}
}

template <>
void rand_uniform<double> (VARR dest, std::vector<VARR> srcs)
{
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape_min = srcs.front().second;
	tensorshape& srcshape_max = srcs.back().second;
	double* d = (double*) dest.first;
	double* s_min = (double*) srcs.front().first;
	double* s_max = (double*) srcs.front().first;
	bool min_mul = srcshape_min.n_elems() > 1;
	bool max_mul = srcshape_max.n_elems() > 1;
	size_t n = destshape.n_elems();

	for (size_t i = 0; i < n; ++i)
	{
		std::uniform_real_distribution<double> dist(s_min[i * min_mul], s_max[i * max_mul]);
		d[i] = dist(nnutils::get_generator());
	}
}

}

#endif /* TENNCOR_DATA_OP_HPP */
