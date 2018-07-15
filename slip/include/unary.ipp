//
//  unary.ipp
//  slip
//

#include <list>

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
T* safe_get (mold::StateRange& state)
{
	char* out = state.get();
	if (nullptr == out)
	{
		throw std::exception();
	}
	return (T*) out;
}

template <typename T>
void copyover (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	clay::Shape& destshape = dest.shape_;
	clay::Shape srcshape = srcs.front().shape();
	size_t n = srcshape.n_elems();
	assert(destshape.n_elems() == n);
	T* d = safe_get<T>(dest);
	const T* s = safe_get<const T>(srcs.front());
	std::memcpy(d, s, sizeof(T) * n);
}

template <typename T>
void unary (clay::State& dest, std::vector<mold::StateRange> srcs,
	std::function<T(const T&)> f)
{
	clay::Shape srcshape = srcs.front().shape();
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
void abs (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	unary<T>(dest, srcs, [](const T& src) { return std::abs(src); });
}

template <typename T>
void neg (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	unary<T>(dest, srcs, [](const T& src) { return -src; });
}

template <typename T>
void logic_not (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	unary<T>(dest, srcs, [](const T& src) { return !src; });
}

template <typename T>
void sin (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	unary<T>(dest, srcs, [](const T& src) { return std::sin(src); });
}

template <typename T>
void cos (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	unary<T>(dest, srcs, [](const T& src) { return std::cos(src); });
}

template <typename T>
void tan (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	unary<T>(dest, srcs, [](const T& src) { return std::tan(src); });
}

template <typename T>
void exp (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	unary<T>(dest, srcs, [](const T& src) { return std::exp(src); });
}

template <typename T>
void log (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	unary<T>(dest, srcs, [](const T& src) { return std::log(src); });
}

template <typename T>
void sqrt (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	unary<T>(dest, srcs, [](const T& src) { return std::sqrt(src); });
}

template <typename T>
void round (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	unary<T>(dest, srcs, [](const T& src) { return std::round(src); });
}

template <typename T>
void transpose (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	clay::Shape& destshape = dest.shape_;
	clay::Shape srcshape = srcs.front().shape();
	T* d = safe_get<T>(dest);
	const T* s = safe_get<const T>(srcs.front());
	std::vector<uint64_t> perm;
	if (srcs.size() > 1)
	{
		mold::StateRange& pstate = srcs[1];
		if (pstate.type() != clay::UINT64)
		{
			throw std::exception();
		}
		uint64_t* ptr = safe_get<uint64_t>(pstate);
		perm = std::vector<uint64_t>(ptr, ptr + pstate.shape().n_elems());
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
void flip (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	if (srcs.size() != 2)
	{
		throw std::exception();
	}
	clay::Shape& shape = dest.shape_;
	T* d = safe_get<T>(dest);
	const T* s = safe_get<const T>(srcs.front());
	mold::StateRange& dstate = srcs[1];
	if (dstate.type() != clay::UINT64)
	{
		throw std::exception();
	}
	size_t ndims = dstate.shape().n_elems();
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
void arg_reduce (clay::State& dest, mold::StateRange& src,
	std::function<bool(const T&,const T&)> cmp)
{
	T* d = safe_get<T>(dest);
	const T* s = safe_get<const T>(src);
	clay::Shape out1 = src.front();
	clay::Shape inner = src.inner();
	clay::Shape out2 = src.back();
	size_t nouter = std::max<size_t>(1, out1.n_elems());
	size_t ninner = inner.n_elems();
	size_t nout2 = std::max<size_t>(1, out2.n_elems());
	assert(ninner > 0);

	for (size_t i = 0; i < nouter; ++i)
	{
		for (size_t j = 0; j < nout2; ++j)
		{
			size_t outidx = i + j * nouter;
			size_t inidx = i + j * ninner * nouter;
			size_t n = i + (j + 1) * ninner * nouter;
			size_t out = inidx;
			for (inidx += nouter; inidx < n; inidx += nouter)
			{
				if (cmp(s[out], s[inidx]))
				{
					out = inidx;
				}
			}
			d[outidx] = out;
		}
	}
}

template <typename T>
void arg_max (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	auto lt = [](const T& a, const T& b)
	{
		return a < b;
	};
	arg_reduce<T>(dest, srcs[0], lt);
}

template <typename T>
void is_max (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	mold::StateRange& src = srcs[0];
	T* d = safe_get<T>(dest);
	const T* s = safe_get<const T>(src);
	clay::Shape out1 = src.front();
	clay::Shape inner = src.inner();
	clay::Shape out2 = src.back();
	size_t nouter = std::max<size_t>(1, out1.n_elems());
	size_t ninner = inner.n_elems();
	size_t nout2 = std::max<size_t>(1, out2.n_elems());
	assert(ninner > 0);

	std::list<size_t> ones;
	for (size_t i = 0; i < nouter; ++i)
	{
		for (size_t j = 0; j < nout2; ++j)
		{
			size_t inidx = i + j * ninner * nouter;
			size_t n = i + (j + 1) * ninner * nouter;
			std::list<size_t> indices = {inidx};
			for (inidx += nouter; inidx < n; inidx += nouter)
			{
				if (s[indices.front()] == s[inidx])
				{
					indices.push_back(inidx);
				}
				else if (s[indices.front()] < s[inidx])
				{
					indices = {inidx};
				}
			}
			ones.insert(ones.end(), indices.begin(), indices.end());
		}
	}
	memset(d, 0, sizeof(T) * dest.shape_.n_elems());
	for (size_t one : ones)
	{
		d[one] = 1;
	}
}

template <typename T>
void reduce (clay::State& dest, mold::StateRange& src,
	std::function<void(T&,const T&)> accum)
{
	T* d = safe_get<T>(dest);
	const T* s = safe_get<const T>(src);
	clay::Shape out1 = src.front();
	clay::Shape inner = src.inner();
	clay::Shape out2 = src.back();
	size_t nouter = std::max<size_t>(1, out1.n_elems());
	size_t ninner = inner.n_elems();
	size_t nout2 = std::max<size_t>(1, out2.n_elems());
	assert(ninner > 0);

	for (size_t i = 0; i < nouter; ++i)
	{
		for (size_t j = 0; j < nout2; ++j)
		{
			size_t outidx = i + j * nouter;
			size_t inidx = i + j * ninner * nouter;
			size_t n = i + (j + 1) * ninner * nouter;
			T out = s[inidx];
			for (inidx += nouter; inidx < n; inidx += nouter)
			{
				accum(out, s[inidx]);
			}
			d[outidx] = out;
		}
	}
}

template <typename T>
void rmax_helper (T& acc, const T& e)
{
	if (acc < e)
	{
		acc = e;
	}
}

template <typename T>
void rmax (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	reduce<T>(dest, srcs[0], rmax_helper<T>);
}

template <typename T>
void rsum_helper (T& acc, const T& e)
{
	acc += e;
}

template <typename T>
void rsum (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	reduce<T>(dest, srcs[0], rsum_helper<T>);
}

}

#endif
