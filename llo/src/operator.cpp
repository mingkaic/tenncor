#include "llo/operator.hpp"

#ifdef LLO_OPERATOR_HPP

template <>
void abs<uint8_t> (std::vector<T>& out, const std::vector<T>& in)
{
	size_t n = out.size();
	assert(n == in.size());
	std::memcpy(&out[0], &in[0], sizeof(uint8_t) * n);
}

template <>
void abs<uint16_t> (std::vector<T>& out, const std::vector<T>& in)
{
	size_t n = out.size();
	assert(n == in.size());
	std::memcpy(&out[0], &in[0], sizeof(uint16_t) * n);
}

template <>
void abs<uint32_t> (std::vector<T>& out, const std::vector<T>& in)
{
	size_t n = out.size();
	assert(n == in.size());
	std::memcpy(&out[0], &in[0], sizeof(uint32_t) * n);
}

template <>
void abs<uint64_t> (std::vector<T>& out, const std::vector<T>& in)
{
	size_t n = out.size();
	assert(n == in.size());
	std::memcpy(&out[0], &in[0], sizeof(uint64_t) * n);
}

template <>
void neg<uint8_t> (std::vector<T>& out, const std::vector<T>& in)
{
	throw std::bad_function_call();
}

template <>
void neg<uint16_t> (std::vector<T>& out, const std::vector<T>& in)
{
	throw std::bad_function_call();
}

template <>
void neg<uint32_t> (std::vector<T>& out, const std::vector<T>& in)
{
	throw std::bad_function_call();
}

template <>
void neg<uint64_t> (std::vector<T>& out, const std::vector<T>& in)
{
	throw std::bad_function_call();
}

template <>
void n_elems<uint64_t> (std::vector<T>& out, const ade::Shape& in)
{
	assert(out.size() == 1);
	out[0] = in.n_elems();
}

template <>
void n_dims<uint8_t> (std::vector<T>& out, const ade::Shape& in, uint8_t dim)
{
	assert(out.size() == 1);
	out[0] = in.at(dim);
}

template <>
void rand_normal<float> (std::vector<float>& out,
	const std::vector<float>& a, const std::vector<float>& b)
{
	binary<float>(dest, a, b,
	[](const float& a, const float& b) -> float
	{
		std::normal_distribution<float> dist(a, b);
		return dist(get_engine());
	});
}

template <>
void rand_normal<double> (std::vector<double>& out,
	const std::vector<double>& a, const std::vector<double>& b)
{
	binary<double>(dest, a, b,
	[](const double& a, const double& b) -> double
	{
		std::normal_distribution<double> dist(a, b);
		return dist(get_engine());
	});
}

#endif
