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
	T* d = safe_get<T*>(dest.data_);
	const T* s = safe_get<const T*>(srcs.front().data_);
	size_t n = dest.shape_.n_elems();
	std::memcpy(d, s, sizeof(uint8_t) * n);
}

template <>
void abs<uint16_t> (clay::State& dest, std::vector<clay::State> srcs)
{
	T* d = safe_get<T*>(dest.data_);
	const T* s = safe_get<const T*>(srcs.front().data_);
	size_t n = dest.shape_.n_elems();
	std::memcpy(d, s, sizeof(uint16_t) * n);
}

template <>
void abs<uint32_t> (clay::State& dest, std::vector<clay::State> srcs)
{
	T* d = safe_get<T*>(dest.data_);
	const T* s = safe_get<const T*>(srcs.front().data_);
	size_t n = dest.shape_.n_elems();
	std::memcpy(d, s, sizeof(uint32_t) * n);
}

template <>
void abs<uint64_t> (clay::State& dest, std::vector<clay::State> srcs)
{
	T* d = safe_get<T*>(dest.data_);
	const T* s = safe_get<const T*>(srcs.front().data_);
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
void rand_binom<float> (VARR_T dest, std::vector<CVAR_T> srcs)
{
	throw std::bad_function_call();
}

template <>
void rand_binom<double> (VARR_T dest, std::vector<CVAR_T> srcs)
{
	throw std::bad_function_call(); 
}

}

#endif
