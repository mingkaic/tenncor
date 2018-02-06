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
void abs (VARR dest, std::vector<VARR> srcs, ARGS)
{
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcs.front().second);
	T* d = dest.first;
	T* s = srcs.front().first;
	size_t n = dest.second.n_elems();
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = std::abs(s[i]);
	}
}

template <typename T>
void neg (VARR dest, std::vector<VARR> srcs, ARGS)
{
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcs.front().second);
	T* d = dest.first;
	T* s = srcs.front().first;
	size_t n = dest.second.n_elems();
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = -stds[i];
	}
}

template <typename T>
void sin (VARR dest, std::vector<VARR> srcs, ARGS)
{
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcs.front().second);
	T* d = dest.first;
	T* s = srcs.front().first;
	size_t n = dest.second.n_elems();
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = std::sin(s[i]);
	}
}

template <typename T>
void cos (VARR dest, std::vector<VARR> srcs, ARGS)
{
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcs.front().second);
	T* d = dest.first;
	T* s = srcs.front().first;
	size_t n = dest.second.n_elems();
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = std::cos(s[i]);
	}
}

template <typename T>
void tan (VARR dest, std::vector<VARR> srcs, ARGS)
{
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcs.front().second);
	T* d = dest.first;
	T* s = srcs.front().first;
	size_t n = dest.second.n_elems();
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = std::tan(s[i]);
	}
}

template <typename T>
void csc (VARR dest, std::vector<VARR> srcs, ARGS)
{
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcs.front().second);
	T* d = dest.first;
	T* s = srcs.front().first;
	size_t n = dest.second.n_elems();
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = 1 / std::sins[i]);
	}
}

template <typename T>
void sec (VARR dest, std::vector<VARR> srcs, ARGS)
{
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcs.front().second);
	T* d = dest.first;
	T* s = srcs.front().first;
	size_t n = dest.second.n_elems();
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = 1 / std::cos(s[i]);
	}
}

template <typename T>
void cot (VARR dest, std::vector<VARR> srcs, ARGS)
{
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcs.front().second);
	T* d = dest.first;
	T* s = srcs.front().first;
	size_t n = dest.second.n_elems();
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = std::cos(s[i]) / std::sin(s[i]);
	}
}

template <typename T>
void exp (VARR dest, std::vector<VARR> srcs, ARGS)
{
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcs.front().second);
	T* d = dest.first;
	T* s = srcs.front().first;
	size_t n = dest.second.n_elems();
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = std::exp(s[i]);
	}
}

template <typename T>
void ln (VARR dest, std::vector<VARR> srcs, ARGS)
{
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcs.front().second);
	T* d = dest.first;
	T* s = srcs.front().first;
	size_t n = dest.second.n_elems();
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = std::log(s[i]);
	}
}

template <typename T>
void sqrt (VARR dest, std::vector<VARR> srcs, ARGS)
{
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcs.front().second);
	T* d = dest.first;
	T* s = srcs.front().first;
	size_t n = dest.second.n_elems();
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = std::sqrt(s[i]);
	}
}

template <typename T>
void round (VARR dest, std::vector<VARR> srcs, ARGS)
{
	// assert(srcs.size() == 1 && dest.second.compatible_with(srcs.front().second);
	T* d = dest.first;
	T* s = srcs.front().first;
	size_t n = dest.second.n_elems();
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = std::round(s[i]);
	}
}

}

#endif
