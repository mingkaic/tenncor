#include <cmath>

#include "internal/eigen/decimal.hpp"

#include <iostream>

#ifdef NUMBERS_DECIMAL_HPP

namespace numbers
{

static inline uint8_t highest_1bit (uint64_t val)
{
	uint8_t r = 0;
	while (val >>= 1)
	{
		r++;
	}
	return r;
}

static inline bool mul_is_safe (uint64_t a, uint64_t b)
{
    auto a_bits = highest_1bit(a);
	auto b_bits = highest_1bit(b);
    return a_bits + b_bits <= 64;
}

Fraction operator * (const Fraction& l, const Fraction& r)
{
	double num;
	uint64_t denom;
    auto proj_num = l.num_ * r.num_;
    if (false == std::isfinite(proj_num) || proj_num > 1000000)
    {
        denom = r.denom_;
        num = double(l);
    }
	else if (mul_is_safe(l.denom_, r.denom_))
	{
		denom = l.denom_ * r.denom_;
		num = l.num_;
	}
	else
	{
		denom = l.denom_;
		num = double(l);
	}
	num *= r.num_;
	return Fraction(num, denom);
}

// same thing as 1 - fraction
Fraction reverse (const Fraction& fraction)
{
    return Fraction(fraction.denom_ - fraction.num_, fraction.denom_);
}

Fraction pow (const Fraction& l, uint64_t r)
{
	if (r == 0)
	{
		return Fraction(1, 1);
	}
	if (r == 1)
	{
		return l;
	}
	auto half = pow(l, r / 2);
	auto out = half * half;
	if (r % 2)
	{
		return out * l;
	}
	return out;
}

}

#endif
