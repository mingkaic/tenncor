//
//  d_nnary.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2018-01-19.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#ifdef TENNCOR_D_NNARY_HPP

namespace nnet
{

template <typename T>
void pow (VARR_T dest, std::vector<CVAR_T> srcs)
{
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape0 = srcs.front().second;
	tensorshape& srcshape1 = srcs.back().second;
	T* d = (T*) dest.first;
	T* bn = (T*) srcs.front().first;
	T* xp = (T*) srcs.back().first;
	bool left_mul = srcshape0.n_elems() > 1;
	bool right_mul = srcshape1.n_elems() > 1;
	size_t n = destshape.n_elems();

	for (size_t i = 0; i < n; ++i)
	{
		d[i] = std::pow(bn[i * left_mul], xp[i * right_mul]);
	}
}

template <typename T>
void add (VARR_T dest, std::vector<CVAR_T> srcs)
{ // todo: mitigation strategies to prevent error propagation for overflows
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape0 = srcs.front().second;
	tensorshape& srcshape1 = srcs.back().second;
	T* d = (T*) dest.first;
	const T* sa = (const T*) srcs.front().first;
	const T* sb = (const T*) srcs.back().first;
	bool left_mul = srcshape0.n_elems() > 1;
	bool right_mul = srcshape1.n_elems() > 1;
	size_t n = destshape.n_elems();

	for (size_t i = 0; i < n; ++i)
	{
		d[i] = sa[i * left_mul] + sb[i * right_mul];
	}
}

template <typename T>
void sub (VARR_T dest, std::vector<CVAR_T> srcs)
{ // todo: mitigation strategies to prevent error propagation for underflows
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape0 = srcs.front().second;
	tensorshape& srcshape1 = srcs.back().second;
	T* d = (T*) dest.first;
	const T* sa = (const T*) srcs.front().first;
	const T* sb = (const T*) srcs.back().first;
	bool left_mul = srcshape0.n_elems() > 1;
	bool right_mul = srcshape1.n_elems() > 1;
	size_t n = destshape.n_elems();

	for (size_t i = 0; i < n; ++i)
	{
		d[i] = sa[i * left_mul] - sb[i * right_mul];
	}
}

template <typename T>
void mul (VARR_T dest, std::vector<CVAR_T> srcs)
{
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape0 = srcs.front().second;
	tensorshape& srcshape1 = srcs.back().second;
	T* d = (T*) dest.first;
	const T* sa = (const T*) srcs.front().first;
	const T* sb = (const T*) srcs.back().first;
	bool left_mul = srcshape0.n_elems() > 1;
	bool right_mul = srcshape1.n_elems() > 1;
	size_t n = destshape.n_elems();

	for (size_t i = 0; i < n; ++i)
	{
		d[i] = sa[i * left_mul] * sb[i * right_mul];
	}
}

template <typename T>
void div (VARR_T dest, std::vector<CVAR_T> srcs)
{ // todo: mitigation strategies to prevent error propagation for large/small numerator/denoms
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape0 = srcs.front().second;
	tensorshape& srcshape1 = srcs.back().second;
	T* d = (T*) dest.first;
	const T* sa = (const T*) srcs.front().first;
	const T* sb = (const T*) srcs.back().first;
	bool left_mul = srcshape0.n_elems() > 1;
	bool right_mul = srcshape1.n_elems() > 1;
	size_t n = destshape.n_elems();

	for (size_t i = 0; i < n; ++i)
	{
		d[i] = sa[i * left_mul] / sb[i * right_mul];
	}
}

template <typename T>
void eq (VARR_T dest, std::vector<CVAR_T> srcs)
{
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape0 = srcs.front().second;
	tensorshape& srcshape1 = srcs.back().second;
	T* d = (T*) dest.first;
	const T* sa = (const T*) srcs.front().first;
	const T* sb = (const T*) srcs.back().first;
	bool left_mul = srcshape0.n_elems() > 1;
	bool right_mul = srcshape1.n_elems() > 1;
	size_t n = destshape.n_elems();

	for (size_t i = 0; i < n; ++i)
	{
		d[i] = (T) sa[i * left_mul] == sb[i * right_mul];
	}
}

template <typename T>
void neq (VARR_T dest, std::vector<CVAR_T> srcs)
{
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape0 = srcs.front().second;
	tensorshape& srcshape1 = srcs.back().second;
	T* d = (T*) dest.first;
	const T* sa = (const T*) srcs.front().first;
	const T* sb = (const T*) srcs.back().first;
	bool left_mul = srcshape0.n_elems() > 1;
	bool right_mul = srcshape1.n_elems() > 1;
	size_t n = destshape.n_elems();

	for (size_t i = 0; i < n; ++i)
	{
		d[i] = (T)  sa[i * left_mul] != sb[i * right_mul];
	}
}

template <typename T>
void lt (VARR_T dest, std::vector<CVAR_T> srcs)
{
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape0 = srcs.front().second;
	tensorshape& srcshape1 = srcs.back().second;
	T* d = (T*) dest.first;
	const T* sa = (const T*) srcs.front().first;
	const T* sb = (const T*) srcs.back().first;
	bool left_mul = srcshape0.n_elems() > 1;
	bool right_mul = srcshape1.n_elems() > 1;
	size_t n = destshape.n_elems();

	for (size_t i = 0; i < n; ++i)
	{
		d[i] = (T)  sa[i * left_mul] < sb[i * right_mul];
	}
}

template <typename T>
void gt (VARR_T dest, std::vector<CVAR_T> srcs)
{
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape0 = srcs.front().second;
	tensorshape& srcshape1 = srcs.back().second;
	T* d = (T*) dest.first;
	const T* sa = (const T*) srcs.front().first;
	const T* sb = (const T*) srcs.back().first;
	bool left_mul = srcshape0.n_elems() > 1;
	bool right_mul = srcshape1.n_elems() > 1;
	size_t n = destshape.n_elems();

	for (size_t i = 0; i < n; ++i)
	{
		d[i] = (T)  sa[i * left_mul] > sb[i * right_mul];
	}
}

template <typename T>
void rand_binom (VARR_T dest, std::vector<CVAR_T> srcs)
{
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape0 = srcs.front().second;
	tensorshape& srcshape1 = srcs.back().second;
	T* d = (T*) dest.first;
	const T* sn = (const T*) srcs.front().first;
	const T* sp = (const T*) srcs.back().first;
	bool left_mul = srcshape0.n_elems() > 1;
	bool right_mul = srcshape1.n_elems() > 1;
	size_t n = destshape.n_elems();

	for (size_t i = 0; i < n; ++i)
	{
		std::binomial_distribution<T> dist(sn[i * left_mul], sp[i * right_mul]);
		d[i] = dist(nnutils::get_generator());
	}
}

template <typename T>
void rand_uniform (VARR_T dest, std::vector<CVAR_T> srcs)
{
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape_min = srcs.front().second;
	tensorshape& srcshape_max = srcs.back().second;
	T* d = (T*) dest.first;
	const T* s_min = (const T*) srcs.front().first;
	const T* s_max = (const T*) srcs.back().first;
	bool min_mul = srcshape_min.n_elems() > 1;
	bool max_mul = srcshape_max.n_elems() > 1;
	size_t n = destshape.n_elems();

	for (size_t i = 0; i < n; ++i)
	{
		std::uniform_int_distribution<T> dist(s_min[i * min_mul], s_max[i * max_mul]);
		d[i] = dist(nnutils::get_generator());
	}
}

template <typename T>
void rand_normal (VARR_T dest, std::vector<CVAR_T> srcs)
{
	throw std::bad_function_call(); // normal distribution with integer types is not acceptable
}

}

#endif
