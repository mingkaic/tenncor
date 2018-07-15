//
//  binary.ipp
//  slip
//

#ifdef SLIP_BINARY_HPP

namespace slip
{

template <typename T>
void binary (clay::State& dest, std::vector<mold::StateRange> srcs,
	std::function<T(const T&,const T&)> f)
{
	clay::Shape& destshape = dest.shape_;
	clay::Shape srcshape0 = srcs.front().shape();
	clay::Shape srcshape1 = srcs.back().shape();
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
void pow (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	binary<T>(dest, srcs, [](const T& b, const T& x) { return std::pow(b, x); });
}

template <typename T>
void add (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	binary<T>(dest, srcs, [](const T& a, const T& b) { return a + b; });
}

template <typename T>
void sub (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	binary<T>(dest, srcs, [](const T& a, const T& b) { return a - b; });
}

template <typename T>
void mul (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	binary<T>(dest, srcs, [](const T& a, const T& b) { return a * b; });
}

template <typename T>
void div (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	binary<T>(dest, srcs, [](const T& a, const T& b) { return a / b; });
}

template <typename T>
void eq (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	binary<T>(dest, srcs, [](const T& a, const T& b) { return a == b; });
}

template <typename T>
void neq (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	binary<T>(dest, srcs, [](const T& a, const T& b) { return a != b; });
}

template <typename T>
void lt (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	binary<T>(dest, srcs, [](const T& a, const T& b) { return a < b; });
}

template <typename T>
void gt (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	binary<T>(dest, srcs, [](const T& a, const T& b) { return a > b; });
}

template <typename T>
void rand_binom (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	clay::Shape& destshape = dest.shape_;
	clay::Shape srcshape0 = srcs.front().shape();
	clay::Shape srcshape1 = srcs.back().shape();
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
void rand_uniform (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	binary<T>(dest, srcs,
	[](const T& a, const T& b)
	{
		std::uniform_int_distribution<T> dist(a, b);
		return dist(get_generator());
	});
}

template <typename T>
void rand_normal (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	throw std::bad_function_call();
}

template <typename T>
void expand (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	if (srcs.size() != 3)
	{
		throw std::exception();
	}
	clay::Shape srcshape = srcs.front().shape();
	T* d = safe_get<T>(dest);
	const T* s = safe_get<const T>(srcs.front());
	mold::StateRange& nstate = srcs[1];
	mold::StateRange& dstate = srcs[2];
	if (nstate.type() != clay::UINT64 || dstate.type() != clay::UINT64)
	{
		throw std::exception();
	}
	if (1 != nstate.shape().n_elems() ||
		1 != dstate.shape().n_elems())
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
