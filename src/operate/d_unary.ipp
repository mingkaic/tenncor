//
//  d_unary.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2018-01-19.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#ifdef TENNCOR_D_UNARY_HPP

namespace nnet
{

template <typename T>
void abs (VARR_T dest, std::vector<CVAR_T> srcs)
{
	tensorshape& srcshape = srcs.front().second;
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcshape);
	T* d = (T*) dest.first;
	const T* s = (const T*) srcs.front().first;
	size_t n = dest.second.n_elems();
	bool src_mul = srcshape.n_elems() > 1;
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = std::abs(s[src_mul * i]);
	}
}

template <typename T>
void neg (VARR_T dest, std::vector<CVAR_T> srcs)
{
	tensorshape& srcshape = srcs.front().second;
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcshape);
	T* d = (T*) dest.first;
	const T* s = (const T*) srcs.front().first;
	size_t n = dest.second.n_elems();
	bool src_mul = srcshape.n_elems() > 1;
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = -s[src_mul * i];
	}
}

template <typename T>
void sin (VARR_T dest, std::vector<CVAR_T> srcs)
{
	tensorshape& srcshape = srcs.front().second;
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcshape);
	T* d = (T*) dest.first;
	const T* s = (const T*) srcs.front().first;
	size_t n = dest.second.n_elems();
	bool src_mul = srcshape.n_elems() > 1;
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = std::sin(s[src_mul * i]);
	}
}

template <typename T>
void cos (VARR_T dest, std::vector<CVAR_T> srcs)
{
	tensorshape& srcshape = srcs.front().second;
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcshape);
	T* d = (T*) dest.first;
	const T* s = (const T*) srcs.front().first;
	size_t n = dest.second.n_elems();
	bool src_mul = srcshape.n_elems() > 1;
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = std::cos(s[src_mul * i]);
	}
}

template <typename T>
void tan (VARR_T dest, std::vector<CVAR_T> srcs)
{
	tensorshape& srcshape = srcs.front().second;
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcshape);
	T* d = (T*) dest.first;
	const T* s = (const T*) srcs.front().first;
	size_t n = dest.second.n_elems();
	bool src_mul = srcshape.n_elems() > 1;
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = std::tan(s[src_mul * i]);
	}
}

template <typename T>
void csc (VARR_T dest, std::vector<CVAR_T> srcs)
{
	tensorshape& srcshape = srcs.front().second;
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcshape);
	T* d = (T*) dest.first;
	const T* s = (const T*) srcs.front().first;
	size_t n = dest.second.n_elems();
	bool src_mul = srcshape.n_elems() > 1;
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = 1 / std::sin(s[src_mul * i]);
	}
}

template <typename T>
void sec (VARR_T dest, std::vector<CVAR_T> srcs)
{
	tensorshape& srcshape = srcs.front().second;
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcshape);
	T* d = (T*) dest.first;
	const T* s = (const T*) srcs.front().first;
	size_t n = dest.second.n_elems();
	bool src_mul = srcshape.n_elems() > 1;
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = 1 / std::cos(s[src_mul * i]);
	}
}

template <typename T>
void cot (VARR_T dest, std::vector<CVAR_T> srcs)
{
	tensorshape& srcshape = srcs.front().second;
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcshape);
	T* d = (T*) dest.first;
	const T* s = (const T*) srcs.front().first;
	size_t n = dest.second.n_elems();
	bool src_mul = srcshape.n_elems() > 1;
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = std::cos(s[src_mul * i]) / std::sin(s[src_mul * i]);
	}
}

template <typename T>
void exp (VARR_T dest, std::vector<CVAR_T> srcs)
{
	tensorshape& srcshape = srcs.front().second;
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcshape);
	T* d = (T*) dest.first;
	const T* s = (const T*) srcs.front().first;
	size_t n = dest.second.n_elems();
	bool src_mul = srcshape.n_elems() > 1;
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = std::exp(s[src_mul * i]);
	}
}

template <typename T>
void log (VARR_T dest, std::vector<CVAR_T> srcs)
{
	tensorshape& srcshape = srcs.front().second;
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcshape);
	T* d = (T*) dest.first;
	const T* s = (const T*) srcs.front().first;
	size_t n = dest.second.n_elems();
	bool src_mul = srcshape.n_elems() > 1;
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = std::log(s[src_mul * i]);
	}
}

template <typename T>
void sqrt (VARR_T dest, std::vector<CVAR_T> srcs)
{
	tensorshape& srcshape = srcs.front().second;
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcshape);
	T* d = (T*) dest.first;
	const T* s = (const T*) srcs.front().first;
	size_t n = dest.second.n_elems();
	bool src_mul = srcshape.n_elems() > 1;
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = std::sqrt(s[src_mul * i]);
	}
}

template <typename T>
void round (VARR_T dest, std::vector<CVAR_T> srcs)
{
	tensorshape& srcshape = srcs.front().second;
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcshape);
	T* d = (T*) dest.first;
	const T* s = (const T*) srcs.front().first;
	size_t n = dest.second.n_elems();
	bool src_mul = srcshape.n_elems() > 1;
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = std::round(s[src_mul * i]);
	}
}

}

#endif
