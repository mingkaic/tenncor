#include "llo/operator.hpp"

#ifdef LLO_OPERATOR_HPP

template <>
void abs<uint8_t> (uint8_t* out, const uint8_t* in, size_t n)
{
	std::memcpy(&out[0], &in[0], sizeof(uint8_t) * n);
}

template <>
void abs<uint16_t> (uint16_t* out, const uint16_t* in, size_t n)
{
	std::memcpy(&out[0], &in[0], sizeof(uint16_t) * n);
}

template <>
void abs<uint32_t> (uint32_t* out, const uint32_t* in, size_t n)
{
	std::memcpy(&out[0], &in[0], sizeof(uint32_t) * n);
}

template <>
void abs<uint64_t> (uint64_t* out, const uint64_t* in, size_t n)
{
	std::memcpy(&out[0], &in[0], sizeof(uint64_t) * n);
}

template <>
void neg<uint8_t> (uint8_t* out, const uint8_t* in, size_t n)
{
	throw std::bad_function_call();
}

template <>
void neg<uint16_t> (uint16_t* out, const uint16_t* in, size_t n)
{
	throw std::bad_function_call();
}

template <>
void neg<uint32_t> (uint32_t* out, const uint32_t* in, size_t n)
{
	throw std::bad_function_call();
}

template <>
void neg<uint64_t> (uint64_t* out, const uint64_t* in, size_t n)
{
	throw std::bad_function_call();
}

template <>
void rand_binom<double> (double* out,
	const double* a, size_t an, const double* b, size_t bn)
{
	size_t n = std::max(an, bn);
	for (size_t i = 0; i < n; ++i)
	{
		std::binomial_distribution<int64_t> dist(a[i % an], b[i % bn]);
		out[i] = dist(util::get_engine());
	}
}

template <>
void rand_binom<float> (float* out,
	const float* a, size_t an, const double* b, size_t bn)
{
	size_t n = std::max(an, bn);
	for (size_t i = 0; i < n; ++i)
	{
		std::binomial_distribution<int32_t> dist(a[i % an], b[i % bn]);
		out[i] = dist(util::get_engine());
	}
}

template <>
void rand_uniform<double> (double* out,
	const double* a, size_t an, const double* b, size_t bn)
{
	binary<double>(out, a, an, b, bn,
	[](const double& a, const double& b)
	{
		std::uniform_real_distribution<double> dist(a, b);
		return dist(util::get_engine());
	});
}

template <>
void rand_uniform<float> (float* out,
	const float* a, size_t an, const float* b, size_t bn)
{
	binary<float>(out, a, an, b, bn,
	[](const float& a, const float& b)
	{
		std::uniform_real_distribution<float> dist(a, b);
		return dist(util::get_engine());
	});
}

template <>
void rand_normal<float> (float* out,
	const float* a, size_t an, const float* b, size_t bn)
{
	binary<float>(out, a, an, b, bn,
	[](const float& a, const float& b) -> float
	{
		std::normal_distribution<float> dist(a, b);
		return dist(util::get_engine());
	});
}

template <>
void rand_normal<double> (double* out,
	const double* a, size_t an, const double* b, size_t bn)
{
	binary<double>(out, a, an, b, bn,
	[](const double& a, const double& b) -> double
	{
		std::normal_distribution<double> dist(a, b);
		return dist(util::get_engine());
	});
}

#endif
