//
//  operations.cpp
//  slip
//

#include "slip/include/operations.hpp"
#include "slip/error.hpp"

#include "clay/error.hpp"

#ifdef SLIP_OPERATIONS_HPP

namespace slip
{

template <>
void abs<uint8_t> (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	uint8_t* d = safe_get<uint8_t>(dest);
	const uint8_t* s = safe_get<const uint8_t>(srcs.front());
	size_t n = dest.shape_.n_elems();
	std::memcpy(d, s, sizeof(uint8_t) * n);
}

template <>
void abs<uint16_t> (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	uint16_t* d = safe_get<uint16_t>(dest);
	const uint16_t* s = safe_get<const uint16_t>(srcs.front());
	size_t n = dest.shape_.n_elems();
	std::memcpy(d, s, sizeof(uint16_t) * n);
}

template <>
void abs<uint32_t> (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	uint32_t* d = safe_get<uint32_t>(dest);
	const uint32_t* s = safe_get<const uint32_t>(srcs.front());
	size_t n = dest.shape_.n_elems();
	std::memcpy(d, s, sizeof(uint32_t) * n);
}

template <>
void abs<uint64_t> (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	uint64_t* d = safe_get<uint64_t>(dest);
	const uint64_t* s = safe_get<const uint64_t>(srcs.front());
	size_t n = dest.shape_.n_elems();
	std::memcpy(d, s, sizeof(uint64_t) * n);
}

template <>
void neg<uint8_t> (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	throw std::bad_function_call();
}

template <>
void neg<uint16_t> (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	throw std::bad_function_call();
}

template <>
void neg<uint32_t> (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	throw std::bad_function_call();
}

template <>
void neg<uint64_t> (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	throw std::bad_function_call();
}

template <>
void rand_binom<float> (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	throw std::bad_function_call();
}

template <>
void rand_binom<double> (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	throw std::bad_function_call();
}

template <>
void rand_uniform<float> (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	binary<float>(dest, srcs,
	[](const float& a, const float& b) -> float
	{
		std::uniform_real_distribution<float> dist(a, b);
		return dist(get_generator());
	});
}

template <>
void rand_uniform<double> (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	binary<double>(dest, srcs,
	[](const double& a, const double& b) -> double
	{
		std::uniform_real_distribution<double> dist(a, b);
		return dist(get_generator());
	});
}

template <>
void rand_normal<float> (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	binary<float>(dest, srcs,
	[](const float& a, const float& b) -> float
	{
		std::normal_distribution<float> dist(a, b);
		return dist(get_generator());
	});
}

template <>
void rand_normal<double> (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	binary<double>(dest, srcs,
	[](const double& a, const double& b) -> double
	{
		std::normal_distribution<double> dist(a, b);
		return dist(get_generator());
	});
}

void n_elems (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	clay::Shape srcshape = srcs.front().shape();
	uint64_t* d = safe_get<uint64_t>(dest);
	d[0] = srcshape.n_elems();
}

void n_dims (clay::State& dest, std::vector<mold::StateRange> srcs)
{
	if (srcs.size() != 2)
	{
		throw BadNArgsError(2, srcs.size());
	}
	clay::Shape srcshape = srcs.front().shape();
	uint64_t* d = safe_get<uint64_t>(dest);
	mold::StateRange& dstate = srcs[1];
	if (dstate.type() != clay::UINT64)
	{
		throw clay::UnsupportedTypeError(dstate.type());
	}
	if (1 != dstate.shape().n_elems())
	{
		throw ShapeMismatchError(clay::Shape({1}), dstate.shape());
	}
	uint64_t dim = *(safe_get<uint64_t>(dstate));
	if (dim < srcshape.rank())
	{
		d[0] = srcshape.at(dim);
	}
	else
	{
		d[0] = 0;
	}
}

}

#endif
