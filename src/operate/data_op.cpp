//
//  data_op.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2018-01-14.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/operate/data_op.hpp"

#ifdef TENNCOR_DATA_OP_HPP

namespace nnet
{

template <typename VALUE_F>
using TYPEMAP_T = std::unordered_map<TENS_TYPE,VALUE_F>;

#define REGISTER_FUNC(FUNC, FUNCTYPE) {#FUNC, TYPEMAP_T<FUNCTYPE>{\
{DOUBLE, nnet::FUNC<double>},{FLOAT, nnet::FUNC<float>},\
{INT8, nnet::FUNC<int8_t>},{UINT8, nnet::FUNC<uint8_t>},\
{INT16, nnet::FUNC<int16_t>},{UINT16, nnet::FUNC<uint16_t>},\
{INT32, nnet::FUNC<int32_t>},{UINT32, nnet::FUNC<uint32_t>},\
{INT64, nnet::FUNC<int64_t>},{UINT64, nnet::FUNC<uint64_t>}}},

#define REG_OPFUNC(FUNC) REGISTER_FUNC(FUNC, VFUNC_F)

#define REG_AGFUNC(FUNC) REGISTER_FUNC(FUNC, AFUNC_F)

static const std::unordered_map<std::string,TYPEMAP_T<VFUNC_F> > ele_registry = {
	REG_OPFUNC(abs)
	REG_OPFUNC(neg)
	REG_OPFUNC(logic_not)
	REG_OPFUNC(sin)
	REG_OPFUNC(cos)
	REG_OPFUNC(tan)
	REG_OPFUNC(exp)
	REG_OPFUNC(log)
	REG_OPFUNC(sqrt)
	REG_OPFUNC(round)

	REG_OPFUNC(pow)
	REG_OPFUNC(add)
	REG_OPFUNC(sub)
	REG_OPFUNC(mul)
	REG_OPFUNC(div)
	REG_OPFUNC(eq)
	REG_OPFUNC(neq)
	REG_OPFUNC(lt)
	REG_OPFUNC(gt)
	REG_OPFUNC(rand_binom)
	REG_OPFUNC(rand_uniform)
	REG_OPFUNC(rand_normal)

	REG_OPFUNC(matmul)
};

static const std::unordered_map<std::string,TYPEMAP_T<AFUNC_F> > agg_registry = {
	REG_AGFUNC(argmax)
	REG_AGFUNC(max)
	REG_AGFUNC(sum)
};

bool has_ele (std::string opname)
{
	return ele_registry.end() != ele_registry.find(opname);
}

bool has_agg (std::string opname)
{
	return agg_registry.end() != agg_registry.find(opname);
}

void ele_op (std::string opname, TENS_TYPE type, VARR_T dest, std::vector<CVAR_T> src)
{
	auto type_it = ele_registry.find(opname);
	if (ele_registry.end() != type_it)
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

void agg_op (std::string opname, TENS_TYPE type, size_t i, void* accum, void* arr)
{
	auto type_it = agg_registry.find(opname);
	if (agg_registry.end() != type_it)
	{
		auto& type_map = type_it->second;
		auto it = type_map.find(type);
		if (type_map.end() != it)
		{
			(it->second)(i, accum, arr);
			return;
		}
	}
	throw std::bad_function_call();
}

template <>
void abs<uint8_t> (VARR_T dest, std::vector<CVAR_T> srcs)
{
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcshape);
	size_t n = dest.second.n_elems();
	std::memcpy(dest.first, srcs.front().first, sizeof(uint8_t) * n);
}

template <>
void abs<uint16_t> (VARR_T dest, std::vector<CVAR_T> srcs)
{
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcshape);
	size_t n = dest.second.n_elems();
	std::memcpy(dest.first, srcs.front().first, sizeof(uint16_t) * n);
}

template <>
void abs<uint32_t> (VARR_T dest, std::vector<CVAR_T> srcs)
{
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcshape);
	size_t n = dest.second.n_elems();
	std::memcpy(dest.first, srcs.front().first, sizeof(uint32_t) * n);
}

template <>
void abs<uint64_t> (VARR_T dest, std::vector<CVAR_T> srcs)
{
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcshape);
	size_t n = dest.second.n_elems();
	std::memcpy(dest.first, srcs.front().first, sizeof(uint64_t) * n);
}

template <>
void neg<uint8_t> (VARR_T, std::vector<CVAR_T>) { throw std::bad_function_call(); }

template <>
void neg<uint16_t> (VARR_T, std::vector<CVAR_T>) { throw std::bad_function_call(); }

template <>
void neg<uint32_t> (VARR_T, std::vector<CVAR_T>) { throw std::bad_function_call(); }

template <>
void neg<uint64_t> (VARR_T, std::vector<CVAR_T>) { throw std::bad_function_call(); }

template <>
void rand_uniform<float> (VARR_T dest, std::vector<CVAR_T> srcs)
{
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape_min = srcs.front().second;
	tensorshape& srcshape_max = srcs.back().second;
	float* d = (float*) dest.first;
	const float* s_min = (const float*) srcs.front().first;
	const float* s_max = (const float*) srcs.back().first;
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
void rand_uniform<double> (VARR_T dest, std::vector<CVAR_T> srcs)
{
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape_min = srcs.front().second;
	tensorshape& srcshape_max = srcs.back().second;
	double* d = (double*) dest.first;
	const double* s_min = (const double*) srcs.front().first;
	const double* s_max = (const double*) srcs.back().first;
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

#endif
