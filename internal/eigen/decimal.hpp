#ifndef NUMBERS_DECIMAL_HPP
#define NUMBERS_DECIMAL_HPP

#include <cstdint>

#include <cassert>

#include "fmts/fmts.hpp"

namespace numbers
{

// assumes num_ <= denom_
struct Fraction final : public fmts::iStringable
{
    Fraction (void) : num_(0), denom_(1) {}

	Fraction (double numerator, uint64_t denominator) :
		num_(numerator), denom_(denominator)
    {
assert(numerator <= (double) denominator);
    }

	operator double() const
	{
		return num_ / denom_;
	}

    std::string to_string (void) const override
    {
        return fmts::sprintf("[%lf](%lf/%u)", (double) *this, num_, denom_);
    }

	double num_;

	uint64_t denom_;
};

Fraction operator * (const Fraction& l, const Fraction& r);

// same thing as 1 - fraction
Fraction reverse (const Fraction& fraction);

Fraction pow (const Fraction& l, uint64_t r);

}

#endif // NUMBERS_DECIMAL_HPP
