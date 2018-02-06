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
void clip (VARR dest, std::vector<VARR> srcs, ARGS)
{
	// assert(srcs.size() < 4 && dest.second.compatible_with(srcs.front().second);
	size_t nargs = srcs.size();
	T min, max;
	if (nargs > 1)
	{
		min = *((T*) srcs[1]);
	}
	if (nargs > 2)
	{
		max = *((T*) srcs[2]);
	}
	if (min > max)
	{
		std::swap(min, max)
	}
	T* d = dest.first;
	T* s = srcs.front().first;
	size_t n = dest.second.n_elems();
	for (size_t i = 0; i < n; ++i)
	{
		if (min > d[i])
		{
			d[i] = min;
		}
		else if (max < d[i])
		{
			d[i] = max;
		}
		else
		{
			d[i] = s[i];
		}
	}
}

template <typename T>
void clip_norm (VARR dest, std::vector<VARR> srcs, ARGS)
{
	assert(srcs.size() == 3);
	// assert(dest.second.compatible_with(srcs.front().second);
	size_t nargs = srcs.size();
	T l2norm = *((T*) srcs[1]);
	T cap = *((T*) srcs[2]);
	T* d = dest.first;
	T* s = srcs.front().first;
	size_t n = dest.second.n_elems();
	if (l2norm > cap)
	{
		for (size_t i = 0; i < n; ++i)
		{
			d[i] = s[i] * cap / l2norm;
		}
	}
	else
	{
		std::memcpy(s, d, sizeof(T) * n);
	}
}

template <typename T>
void binom (VARR dest, std::vector<VARR> srcs, ARGS)
{
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape0 = srcs.front().second;
	tensorshape& srcshape1 = srcs.back().second;
	T* d = dest.first;
	T* sn = srcs.front().first;
	T* sp = srcs.front().first;
	bool left_mul = srcshape0.n_elems() > 1;
	bool right_mul = srcshape1.n_elems() > 1;
	size_t n = destshape.n_elems();

	
	for (size_t i = 0; i < n_out; ++i)
	{
		std::binomial_distribution<int> dist(sn[i * left_mul], sp[i * right_mul])
		d[i] = dist(nnutils::get_generator());
	}
}

template <typename T>
void pow (VARR dest, std::vector<VARR> srcs, ARGS)
{
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape0 = srcs.front().second;
	tensorshape& srcshape1 = srcs.back().second;
	T* d = dest.first;
	T* bn = srcs.front().first;
	T* xp = srcs.front().first;
	bool left_mul = srcshape0.n_elems() > 1;
	bool right_mul = srcshape1.n_elems() > 1;
	size_t n = destshape.n_elems();

	
	for (size_t i = 0; i < n_out; ++i)
	{
		d[i] = std::pow(bn[i * left_mul], xp[i * right_mul]);
	}
}

template <typename T>
void add (VARR dest, std::vector<VARR> srcs, ARGS)
{
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape0 = srcs.front().second;
	tensorshape& srcshape1 = srcs.back().second;
	T* d = dest.first;
	T* sa = srcs.front().first;
	T* sb = srcs.back().first;
	bool left_mul = srcshape0.n_elems() > 1;
	bool right_mul = srcshape1.n_elems() > 1;
	size_t n = destshape.n_elems();

	
	for (size_t i = 0; i < n_out; ++i)
	{
		d[i] = sa[i * left_mul] + sb[i * right_mul];
	}
}

template <typename T>
void sub (VARR dest, std::vector<VARR> srcs, ARGS)
{
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape0 = srcs.front().second;
	tensorshape& srcshape1 = srcs.back().second;
	T* d = dest.first;
	T* sa = srcs.front().first;
	T* sb = srcs.back().first;
	bool left_mul = srcshape0.n_elems() > 1;
	bool right_mul = srcshape1.n_elems() > 1;
	size_t n = destshape.n_elems();

	
	for (size_t i = 0; i < n_out; ++i)
	{
		d[i] = sa[i * left_mul] - sb[i * right_mul];
	}
}

template <typename T>
void mul (VARR dest, std::vector<VARR> srcs, ARGS)
{
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape0 = srcs.front().second;
	tensorshape& srcshape1 = srcs.back().second;
	T* d = dest.first;
	T* sa = srcs.front().first;
	T* sb = srcs.back().first;
	bool left_mul = srcshape0.n_elems() > 1;
	bool right_mul = srcshape1.n_elems() > 1;
	size_t n = destshape.n_elems();

	
	for (size_t i = 0; i < n_out; ++i)
	{
		d[i] = sa[i * left_mul] * sb[i * right_mul];
	}
}

template <typename T>
void div (VARR dest, std::vector<VARR> srcs, ARGS)
{
	// assert(srcs.size() == 2);
	tensorshape& destshape = dest.second;
	tensorshape& srcshape0 = srcs.front().second;
	tensorshape& srcshape1 = srcs.back().second;
	T* d = dest.first;
	T* sa = srcs.front().first;
	T* sb = srcs.back().first;
	bool left_mul = srcshape0.n_elems() > 1;
	bool right_mul = srcshape1.n_elems() > 1;
	size_t n = destshape.n_elems();

	
	for (size_t i = 0; i < n_out; ++i)
	{
		d[i] = sa[i * left_mul] / sb[i * right_mul];
	}
}

}

#endif
