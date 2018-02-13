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
void pow (VARR dest, std::vector<VARR> srcs)
{
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape0 = srcs.front().second;
	tensorshape& srcshape1 = srcs.back().second;
	T* d = (T*) dest.first;
	T* bn = (T*) srcs.front().first;
	T* xp = (T*) srcs.front().first;
	bool left_mul = srcshape0.n_elems() > 1;
	bool right_mul = srcshape1.n_elems() > 1;
	size_t n = destshape.n_elems();

	for (size_t i = 0; i < n; ++i)
	{
		d[i] = std::pow(bn[i * left_mul], xp[i * right_mul]);
	}
}

template <typename T>
void add (VARR dest, std::vector<VARR> srcs)
{
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape0 = srcs.front().second;
	tensorshape& srcshape1 = srcs.back().second;
	T* d = (T*) dest.first;
	T* sa = (T*) srcs.front().first;
	T* sb = (T*) srcs.back().first;
	bool left_mul = srcshape0.n_elems() > 1;
	bool right_mul = srcshape1.n_elems() > 1;
	size_t n = destshape.n_elems();

	for (size_t i = 0; i < n; ++i)
	{
		d[i] = sa[i * left_mul] + sb[i * right_mul];
	}
}

template <typename T>
void sub (VARR dest, std::vector<VARR> srcs)
{
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape0 = srcs.front().second;
	tensorshape& srcshape1 = srcs.back().second;
	T* d = (T*) dest.first;
	T* sa = (T*) srcs.front().first;
	T* sb = (T*) srcs.back().first;
	bool left_mul = srcshape0.n_elems() > 1;
	bool right_mul = srcshape1.n_elems() > 1;
	size_t n = destshape.n_elems();

	for (size_t i = 0; i < n; ++i)
	{
		d[i] = sa[i * left_mul] - sb[i * right_mul];
	}
}

template <typename T>
void mul (VARR dest, std::vector<VARR> srcs)
{
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape0 = srcs.front().second;
	tensorshape& srcshape1 = srcs.back().second;
	T* d = (T*) dest.first;
	T* sa = (T*) srcs.front().first;
	T* sb = (T*) srcs.back().first;
	bool left_mul = srcshape0.n_elems() > 1;
	bool right_mul = srcshape1.n_elems() > 1;
	size_t n = destshape.n_elems();

	for (size_t i = 0; i < n; ++i)
	{
		d[i] = sa[i * left_mul] * sb[i * right_mul];
	}
}

template <typename T>
void div (VARR dest, std::vector<VARR> srcs)
{
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape0 = srcs.front().second;
	tensorshape& srcshape1 = srcs.back().second;
	T* d = (T*) dest.first;
	T* sa = (T*) srcs.front().first;
	T* sb = (T*) srcs.back().first;
	bool left_mul = srcshape0.n_elems() > 1;
	bool right_mul = srcshape1.n_elems() > 1;
	size_t n = destshape.n_elems();

	for (size_t i = 0; i < n; ++i)
	{
		d[i] = sa[i * left_mul] / sb[i * right_mul];
	}
}

template <typename T>
void eq (VARR dest, std::vector<VARR> srcs)
{
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape0 = srcs.front().second;
	tensorshape& srcshape1 = srcs.back().second;
	T* d = (T*) dest.first;
	T* sa = (T*) srcs.front().first;
	T* sb = (T*) srcs.back().first;
	bool left_mul = srcshape0.n_elems() > 1;
	bool right_mul = srcshape1.n_elems() > 1;
	size_t n = destshape.n_elems();

	for (size_t i = 0; i < n; ++i)
	{
		d[i] = (T) sa[i * left_mul] == sb[i * right_mul];
	}
}

template <typename T>
void neq (VARR dest, std::vector<VARR> srcs)
{
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape0 = srcs.front().second;
	tensorshape& srcshape1 = srcs.back().second;
	T* d = (T*) dest.first;
	T* sa = (T*) srcs.front().first;
	T* sb = (T*) srcs.back().first;
	bool left_mul = srcshape0.n_elems() > 1;
	bool right_mul = srcshape1.n_elems() > 1;
	size_t n = destshape.n_elems();

	for (size_t i = 0; i < n; ++i)
	{
		d[i] = (T)  sa[i * left_mul] != sb[i * right_mul];
	}
}

template <typename T>
void lt (VARR dest, std::vector<VARR> srcs)
{
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape0 = srcs.front().second;
	tensorshape& srcshape1 = srcs.back().second;
	T* d = (T*) dest.first;
	T* sa = (T*) srcs.front().first;
	T* sb = (T*) srcs.back().first;
	bool left_mul = srcshape0.n_elems() > 1;
	bool right_mul = srcshape1.n_elems() > 1;
	size_t n = destshape.n_elems();

	for (size_t i = 0; i < n; ++i)
	{
		d[i] = (T)  sa[i * left_mul] < sb[i * right_mul];
	}
}

template <typename T>
void gt (VARR dest, std::vector<VARR> srcs)
{
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape0 = srcs.front().second;
	tensorshape& srcshape1 = srcs.back().second;
	T* d = (T*) dest.first;
	T* sa = (T*) srcs.front().first;
	T* sb = (T*) srcs.back().first;
	bool left_mul = srcshape0.n_elems() > 1;
	bool right_mul = srcshape1.n_elems() > 1;
	size_t n = destshape.n_elems();

	for (size_t i = 0; i < n; ++i)
	{
		d[i] = (T)  sa[i * left_mul] > sb[i * right_mul];
	}
}

template <typename T>
void rand_binom (VARR dest, std::vector<VARR> srcs)
{
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape0 = srcs.front().second;
	tensorshape& srcshape1 = srcs.back().second;
	T* d = (T*) dest.first;
	T* sn = (T*) srcs.front().first;
	T* sp = (T*) srcs.front().first;
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
void rand_uniform (VARR dest, std::vector<VARR> srcs)
{
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape_min = srcs.front().second;
	tensorshape& srcshape_max = srcs.back().second;
	T* d = (T*) dest.first;
	T* s_min = (T*) srcs.front().first;
	T* s_max = (T*) srcs.front().first;
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
void rand_normal (VARR dest, std::vector<VARR> srcs)
{
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape_min = srcs.front().second;
	tensorshape& srcshape_max = srcs.back().second;
	T* d = (T*) dest.first;
	T* s_min = (T*) srcs.front().first;
	T* s_max = (T*) srcs.front().first;
	bool min_mul = srcshape_min.n_elems() > 1;
	bool max_mul = srcshape_max.n_elems() > 1;
	size_t n = destshape.n_elems();

	for (size_t i = 0; i < n; ++i)
	{
		std::normal_distribution<T> dist(s_min[i * min_mul], s_max[i * max_mul]);
		d[i] = dist(nnutils::get_generator());
	}
}

}

#endif
