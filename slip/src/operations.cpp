//
//  operations.cpp
//  slip
//

#include "slip/operations.hpp"

#ifdef SLIP_OPERATIONS_HPP

namespace slip
{

template <>
void abs<uint8_t> (clay::State& dest, std::vector<clay::State> srcs)
{
	uint8_t* d = safe_get<uint8_t>(dest.data_);
	const uint8_t* s = safe_get<const uint8_t>(srcs.front().data_);
	size_t n = dest.shape_.n_elems();
	std::memcpy(d, s, sizeof(uint8_t) * n);
}

template <>
void abs<uint16_t> (clay::State& dest, std::vector<clay::State> srcs)
{
	uint16_t* d = safe_get<uint16_t>(dest.data_);
	const uint16_t* s = safe_get<const uint16_t>(srcs.front().data_);
	size_t n = dest.shape_.n_elems();
	std::memcpy(d, s, sizeof(uint16_t) * n);
}

template <>
void abs<uint32_t> (clay::State& dest, std::vector<clay::State> srcs)
{
	uint32_t* d = safe_get<uint32_t>(dest.data_);
	const uint32_t* s = safe_get<const uint32_t>(srcs.front().data_);
	size_t n = dest.shape_.n_elems();
	std::memcpy(d, s, sizeof(uint32_t) * n);
}

template <>
void abs<uint64_t> (clay::State& dest, std::vector<clay::State> srcs)
{
	uint64_t* d = safe_get<uint64_t>(dest.data_);
	const uint64_t* s = safe_get<const uint64_t>(srcs.front().data_);
	size_t n = dest.shape_.n_elems();
	std::memcpy(d, s, sizeof(uint64_t) * n);
}

template <>
void neg<uint8_t> (clay::State& dest, std::vector<clay::State> srcs)
{
    throw std::bad_function_call();
}

template <>
void neg<uint16_t> (clay::State& dest, std::vector<clay::State> srcs)
{
    throw std::bad_function_call();
}

template <>
void neg<uint32_t> (clay::State& dest, std::vector<clay::State> srcs)
{
    throw std::bad_function_call();
}

template <>
void neg<uint64_t> (clay::State& dest, std::vector<clay::State> srcs)
{
    throw std::bad_function_call();
}

template <>
void rand_binom<float> (clay::State& dest, std::vector<clay::State> srcs)
{
	throw std::bad_function_call();
}

template <>
void rand_binom<double> (clay::State& dest, std::vector<clay::State> srcs)
{
	throw std::bad_function_call(); 
}

template <>
void rand_uniform<float> (clay::State& dest, std::vector<clay::State> srcs)
{
	binary<float>(dest, srcs,
	[](const float& a, const float& b) -> float
	{
		std::uniform_real_distribution<float> dist(a, b);
		return dist(slip::get_generator());
	});
}

template <>
void rand_uniform<double> (clay::State& dest, std::vector<clay::State> srcs)
{
	binary<double>(dest, srcs,
	[](const double& a, const double& b) -> double
	{
		std::uniform_real_distribution<double> dist(a, b);
		return dist(slip::get_generator());
	});
}

template <>
void rand_normal<float> (clay::State& dest, std::vector<clay::State> srcs)
{
	binary<float>(dest, srcs,
	[](const float& a, const float& b) -> float
	{
		std::normal_distribution<float> dist(a, b);
		return dist(slip::get_generator());
	});
}

template <>
void rand_normal<double> (clay::State& dest, std::vector<clay::State> srcs)
{
	binary<double>(dest, srcs,
	[](const double& a, const double& b) -> double
	{
		std::normal_distribution<double> dist(a, b);
		return dist(slip::get_generator());
	});
}

}

#endif
