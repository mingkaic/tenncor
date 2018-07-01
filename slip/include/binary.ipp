//
//  binary.ipp
//  slip
//

#ifdef SLIP_BINARY_HPP

namespace slip
{

template <typename T>
void binary (clay::State& dest, std::vector<clay::State> srcs,
	std::function<T(const T&,const T&)> f)
{
	clay::Shape& destshape = dest.shape_;
	clay::Shape& srcshape0 = srcs.front().shape_;
	clay::Shape& srcshape1 = srcs.back().shape_;
	T* d = safe_get<T>(dest);
	const T* a = safe_get<const T>(srcs.front());
	const T* b = safe_get<const T>(srcs.back());
	bool left_mul = srcshape0.n_elems() > 1;
	bool right_mul = srcshape1.n_elems() > 1;
	size_t n = destshape.n_elems();

	for (size_t i = 0; i < n; ++i)
	{
		d[i] = f(a[i * left_mul], b[i * right_mul]);
	}
}

template <typename T>
void pow (clay::State& dest, std::vector<clay::State> srcs)
{
	binary<T>(dest, srcs, [](const T& b, const T& x) { return std::pow(b, x); });
}

template <typename T>
void add (clay::State& dest, std::vector<clay::State> srcs)
{
	binary<T>(dest, srcs, [](const T& a, const T& b) { return a + b; });
}

template <typename T>
void sub (clay::State& dest, std::vector<clay::State> srcs)
{
	binary<T>(dest, srcs, [](const T& a, const T& b) { return a - b; });
}

template <typename T>
void mul (clay::State& dest, std::vector<clay::State> srcs)
{
	binary<T>(dest, srcs, [](const T& a, const T& b) { return a * b; });
}

template <typename T>
void div (clay::State& dest, std::vector<clay::State> srcs)
{
	binary<T>(dest, srcs, [](const T& a, const T& b) { return a / b; });
}

template <typename T>
void eq (clay::State& dest, std::vector<clay::State> srcs)
{
	binary<T>(dest, srcs, [](const T& a, const T& b) { return a == b; });
}

template <typename T>
void neq (clay::State& dest, std::vector<clay::State> srcs)
{
	binary<T>(dest, srcs, [](const T& a, const T& b) { return a != b; });
}

template <typename T>
void lt (clay::State& dest, std::vector<clay::State> srcs)
{
	binary<T>(dest, srcs, [](const T& a, const T& b) { return a < b; });
}

template <typename T>
void gt (clay::State& dest, std::vector<clay::State> srcs)
{
	binary<T>(dest, srcs, [](const T& a, const T& b) { return a > b; });
}

template <typename T>
void rand_binom (clay::State& dest, std::vector<clay::State> srcs)
{
	clay::Shape& destshape = dest.shape_;
	clay::Shape& srcshape0 = srcs.front().shape_;
	clay::Shape& srcshape1 = srcs.back().shape_;
	T* d = safe_get<T>(dest);
	const T* sn = safe_get<const T>(srcs.front());
	const double* sp = safe_get<const double>(srcs.back());
	bool left_mul = srcshape0.n_elems() > 1;
	bool right_mul = srcshape1.n_elems() > 1;
	size_t n = destshape.n_elems();

	for (size_t i = 0; i < n; ++i)
	{
		std::binomial_distribution<T> dist(sn[i * left_mul], sp[i * right_mul]);
		d[i] = dist(get_generator());
	}
}

template <typename T>
void rand_uniform (clay::State& dest, std::vector<clay::State> srcs)
{
	binary<T>(dest, srcs,
	[](const T& a, const T& b)
	{
		std::uniform_int_distribution<T> dist(a, b);
		return dist(get_generator());
	});
}

template <typename T>
void rand_normal (clay::State& dest, std::vector<clay::State> srcs)
{
	throw std::bad_function_call();
}

template <typename T>
void argmax (clay::State& dest, std::vector<clay::State> srcs)
{
	assert(srcs.size() > 1);
	clay::Shape& srcshape = srcs.front().shape_;
	size_t rank = srcshape.rank();
	if (rank > 1)
	{
		T* d = safe_get<T>(dest);
		const T* s = safe_get<const T>(srcs.front());
		uint64_t dim = *(safe_get<uint64_t>(srcs[1]));
		assert(rank > dim);
		std::vector<size_t> slist = srcshape.as_list();
		slist[dim] = 1;
		clay::Shape nilshape = slist;
		std::vector<size_t> coord;
		size_t n = nilshape.n_elems();
		size_t nd = srcshape.at(dim);
		for (size_t i = 0; i < n; ++i)
		{
			coord = coordinate(nilshape, i);
			d[i] = index(srcshape, coord);
			for (size_t j = 1; j < nd; ++j)
			{
				coord[dim] = j;
				size_t srcidx = index(srcshape, coord);
				if (s[(size_t) d[i]] < s[srcidx])
				{
					d[i] = srcidx;
				}
			}
		}
	}
	else
	{
		unar_argmax<T>(dest, srcs);
	}
}

template <typename T>
void max (clay::State& dest, std::vector<clay::State> srcs)
{
	assert(srcs.size() > 1);
	clay::Shape& srcshape = srcs.front().shape_;
	size_t rank = srcshape.rank();
	if (rank > 1)
	{
		T* d = safe_get<T>(dest);
		const T* s = safe_get<const T>(srcs.front());
		uint64_t dim = *(safe_get<uint64_t>(srcs[1]));
		assert(rank > dim);
		std::vector<size_t> slist = srcshape.as_list();
		slist[dim] = 1;
		clay::Shape nilshape = slist;
		std::vector<size_t> coord;
		size_t n = nilshape.n_elems();
		size_t nd = srcshape.at(dim);
		for (size_t i = 0; i < n; ++i)
		{
			coord = coordinate(nilshape, i);
			d[i] = s[index(srcshape, coord)];
			for (size_t j = 1; j < nd; ++j)
			{
				coord[dim] = j;
				size_t srcidx = index(srcshape, coord);
				if (d[i] < s[srcidx])
				{
					d[i] = s[srcidx];
				}
			}
		}
	}
	else
	{
		unar_max<T>(dest, srcs);
	}
}

template <typename T>
void sum (clay::State& dest, std::vector<clay::State> srcs)
{
	assert(srcs.size() > 1);
	clay::Shape& srcshape = srcs.front().shape_;
	size_t rank = srcshape.rank();
	if (rank > 1)
	{
		T* d = safe_get<T>(dest);
		const T* s = safe_get<const T>(srcs.front());
		uint64_t dim = *(safe_get<uint64_t>(srcs[1]));
		assert(rank > dim);
		std::vector<size_t> slist = srcshape.as_list();
		slist[dim] = 1;
		clay::Shape nilshape = slist;
		std::vector<size_t> coord;
		size_t n = nilshape.n_elems();
		size_t nd = srcshape.at(dim);
		for (size_t i = 0; i < n; ++i)
		{
			coord = coordinate(nilshape, i);
			d[i] = s[index(srcshape, coord)];
			for (size_t j = 1; j < nd; ++j)
			{
				coord[dim] = j;
				size_t srcidx = index(srcshape, coord);
				d[i] += s[srcidx];
			}
		}
	}
	else
	{
		unar_sum<T>(dest, srcs);
	}
}

template <typename T>
void expand (clay::State& dest, std::vector<clay::State> srcs)
{
	if (srcs.size() != 3)
	{
		throw std::exception();
	}
	clay::Shape& srcshape = srcs.front().shape_;
	T* d = safe_get<T>(dest);
	const T* s = safe_get<const T>(srcs.front());
	clay::State& nstate = srcs[1];
	clay::State& dstate = srcs[2];
	if (nstate.dtype_ != clay::UINT64 ||
		dstate.dtype_ != clay::UINT64)
	{
		throw std::exception();
	}
	if (1 != nstate.shape_.n_elems() ||
		1 != dstate.shape_.n_elems())
	{
		throw std::exception();
	}
	uint64_t mul = *(safe_get<uint64_t>(nstate));
	uint64_t dim = *(safe_get<uint64_t>(dstate));
	std::vector<size_t> slist = srcshape.as_list();
	auto it = slist.begin();
	size_t innern = std::accumulate(it, it + dim, 1, std::multiplies<size_t>());
	size_t outern = srcshape.n_elems();
	size_t repeats = outern / innern;
	size_t nexpansion = innern * mul;
	for (size_t j = 0; j < repeats; ++j)
	{
		for (size_t i = 0; i < mul; ++i)
		{
			size_t outidx = j * nexpansion + i * innern;
			size_t inidx = j * innern;
			std::memcpy(d + outidx, s + inidx, innern * sizeof(T));
		}
	}
}

}

#endif /* SLIP_BINARY_HPP */
