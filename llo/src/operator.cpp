#include "llo/operator.hpp"

#ifdef LLO_OPERATOR_HPP

namespace llo
{

EngineT& get_engine (void)
{
	static EngineT engine;
	return engine;
}

template <>
void abs<uint8_t> (uint8_t* out, VecRef<uint8_t> in)
{
	std::memcpy(out, in.data, sizeof(uint8_t) * in.n);
}

template <>
void abs<uint16_t> (uint16_t* out, VecRef<uint16_t> in)
{
	std::memcpy(out, in.data, sizeof(uint16_t) * in.n);
}

template <>
void abs<uint32_t> (uint32_t* out, VecRef<uint32_t> in)
{
	std::memcpy(out, in.data, sizeof(uint32_t) * in.n);
}

template <>
void abs<uint64_t> (uint64_t* out, VecRef<uint64_t> in)
{
	std::memcpy(out, in.data, sizeof(uint64_t) * in.n);
}

template <>
void neg<uint8_t> (uint8_t* out, VecRef<uint8_t> in)
{
	throw std::bad_function_call();
}

template <>
void neg<uint16_t> (uint16_t* out, VecRef<uint16_t> in)
{
	throw std::bad_function_call();
}

template <>
void neg<uint32_t> (uint32_t* out, VecRef<uint32_t> in)
{
	throw std::bad_function_call();
}

template <>
void neg<uint64_t> (uint64_t* out, VecRef<uint64_t> in)
{
	throw std::bad_function_call();
}

template <>
void rand_binom<double> (double* out, VecRef<double> a, VecRef<double> b)
{
	size_t n = std::max(a.n, b.n);
	for (size_t i = 0; i < n; ++i)
	{
		std::binomial_distribution<int64_t> dist(
			a.data[i % a.n], b.data[i % b.n]);
		out[i] = dist(get_engine());
	}
}

template <>
void rand_binom<float> (float* out, VecRef<float> a, VecRef<double> b)
{
	size_t n = std::max(a.n, b.n);
	for (size_t i = 0; i < n; ++i)
	{
		std::binomial_distribution<int32_t> dist(
			a.data[i % a.n], b.data[i % b.n]);
		out[i] = dist(get_engine());
	}
}

template <>
void rand_uniform<double> (double* out, VecRef<double> a, VecRef<double> b)
{
	binary<double>(out, a, b,
	[](const double& a, const double& b)
	{
		std::uniform_real_distribution<double> dist(a, b);
		return dist(get_engine());
	});
}

template <>
void rand_uniform<float> (float* out, VecRef<float> a, VecRef<float> b)
{
	binary<float>(out, a, b,
	[](const float& a, const float& b)
	{
		std::uniform_real_distribution<float> dist(a, b);
		return dist(get_engine());
	});
}

template <>
void rand_normal<float> (float* out, VecRef<float> a, VecRef<float> b)
{
	binary<float>(out, a, b,
	[](const float& a, const float& b) -> float
	{
		std::normal_distribution<float> dist(a, b);
		return dist(get_engine());
	});
}

template <>
void rand_normal<double> (double* out, VecRef<double> a, VecRef<double> b)
{
	binary<double>(out, a, b,
	[](const double& a, const double& b) -> double
	{
		std::normal_distribution<double> dist(a, b);
		return dist(get_engine());
	});
}

}

#endif
