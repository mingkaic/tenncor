//
//  unary.ipp
//  slip
//

#ifdef SLIP_UNARY_HPP

namespace slip
{

template <typename T>
T* safe_get (clay::State& state)
{
	char* out = state.get();
	if (nullptr == out)
	{
		throw std::exception();
	}
	return (T*) out;
}

template <typename T>
void copyover (clay::State& dest, std::vector<clay::State> srcs)
{
	clay::Shape& destshape = dest.shape_;
	clay::Shape& srcshape = srcs.front().shape_;
	size_t n = srcshape.n_elems();
	assert(destshape.n_elems() == n);
	T* d = safe_get<T>(dest);
	const T* s = safe_get<const T>(srcs.front());
	std::memcpy(d, s, sizeof(T) * n);
}

template <typename T>
void unary (clay::State& dest, std::vector<clay::State> srcs,
	std::function<T(const T&)> f)
{
	clay::Shape& srcshape = srcs.front().shape_;
	T* d = safe_get<T>(dest);
	const T* s = safe_get<const T>(srcs.front());
	size_t n = dest.shape_.n_elems();
	bool src_mul = srcshape.n_elems() > 1;
	for (size_t i = 0; i < n; ++i)
	{
		d[i] = f(s[src_mul * i]);
	}
}

template <typename T>
void abs (clay::State& dest, std::vector<clay::State> srcs)
{
	unary<T>(dest, srcs, [](const T& src) { return std::abs(src); });
}

template <typename T>
void neg (clay::State& dest, std::vector<clay::State> srcs)
{
	unary<T>(dest, srcs, [](const T& src) { return -src; });
}

template <typename T>
void logic_not (clay::State& dest, std::vector<clay::State> srcs)
{
	unary<T>(dest, srcs, [](const T& src) { return !src; });
}

template <typename T>
void sin (clay::State& dest, std::vector<clay::State> srcs)
{
	unary<T>(dest, srcs, [](const T& src) { return std::sin(src); });
}

template <typename T>
void cos (clay::State& dest, std::vector<clay::State> srcs)
{
	unary<T>(dest, srcs, [](const T& src) { return std::cos(src); });
}

template <typename T>
void tan (clay::State& dest, std::vector<clay::State> srcs)
{
	unary<T>(dest, srcs, [](const T& src) { return std::tan(src); });
}

template <typename T>
void exp (clay::State& dest, std::vector<clay::State> srcs)
{
	unary<T>(dest, srcs, [](const T& src) { return std::exp(src); });
}

template <typename T>
void log (clay::State& dest, std::vector<clay::State> srcs)
{
	unary<T>(dest, srcs, [](const T& src) { return std::log(src); });
}

template <typename T>
void sqrt (clay::State& dest, std::vector<clay::State> srcs)
{
	unary<T>(dest, srcs, [](const T& src) { return std::sqrt(src); });
}

template <typename T>
void round (clay::State& dest, std::vector<clay::State> srcs)
{
	unary<T>(dest, srcs, [](const T& src) { return std::round(src); });
}

template <typename T>
void transpose (clay::State& dest, std::vector<clay::State> srcs)
{
	clay::Shape& destshape = dest.shape_;
	clay::Shape& srcshape = srcs.front().shape_;
	T* d = safe_get<T>(dest);
	const T* s = safe_get<const T>(srcs.front());
	std::vector<uint64_t> perm;
	if (srcs.size() > 1)
	{
		clay::State& pstate = srcs[1];
		if (pstate.dtype_ != clay::UINT64)
		{
			throw std::exception();
		}
		uint64_t* ptr = safe_get<uint64_t>(pstate);
		perm = std::vector<uint64_t>(ptr, ptr + pstate.shape_.n_elems());
	}
	else
	{
		perm = std::vector<uint64_t>(srcshape.rank());
		std::iota(perm.rbegin(), perm.rend(), 0);
	}
	std::vector<size_t> tmp_coord;
	std::vector<size_t> coord;
	for (size_t i = 0, n = destshape.n_elems();
		i < n; ++i)
	{
		coord = tmp_coord = clay::coordinate(destshape, i);
		for (size_t j = 0, permsize = perm.size();
			j < permsize; ++j)
		{
			coord[perm[j]] = tmp_coord[j];
		}
		d[i] = s[clay::index(srcshape, coord)];
	}
}

template <typename T>
void flip (clay::State& dest, std::vector<clay::State> srcs)
{
	if (srcs.size() != 2)
	{
		throw std::exception();
	}
	clay::Shape& shape = dest.shape_;
	T* d = safe_get<T>(dest);
	const T* s = safe_get<const T>(srcs.front());
	clay::State& dstate = srcs[1];
	if (dstate.dtype_ != clay::UINT64)
	{
		throw std::exception();
	}
	size_t ndims = dstate.shape_.n_elems();
	uint64_t* dims = safe_get<uint64_t>(dstate);
	std::vector<size_t> slist = shape.as_list();
	std::vector<size_t> coord;
	for (size_t i = 0, n = shape.n_elems();
		i < n; ++i)
	{
		coord = clay::coordinate(shape, i);
		for (size_t j = 0; j < ndims; ++j)
		{
			coord[dims[j]] = slist[dims[j]] - coord[dims[j]] - 1;
		}
		d[i] = s[clay::index(shape, coord)];
	}
}

template <typename T>
void unar_argmax (clay::State& dest, std::vector<clay::State> srcs)
{
	clay::Shape& srcshape = srcs.front().shape_;
	T* d = safe_get<T>(dest);
	const T* s = safe_get<const T>(srcs.front());
	size_t n = srcshape.n_elems();
	*d = std::distance(s, std::max_element(s, s + n));
}

template <typename T>
void unar_max (clay::State& dest, std::vector<clay::State> srcs)
{
	clay::Shape& srcshape = srcs.front().shape_;
	T* d = safe_get<T>(dest);
	const T* s = safe_get<const T>(srcs.front());
	size_t n = srcshape.n_elems();
	*d = *(std::max_element(s, s + n));
}

template <typename T>
void unar_sum (clay::State& dest, std::vector<clay::State> srcs)
{
	clay::Shape& srcshape = srcs.front().shape_;
	T* d = safe_get<T>(dest);
	const T* s = safe_get<const T>(srcs.front());
	size_t n = srcshape.n_elems();
	*d = std::accumulate(s, s + n, (T) 0);
}

}

#endif
