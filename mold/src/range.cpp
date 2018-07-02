//
//  range.cpp
//  mold
//

#include "mold/range.hpp"

#ifdef MOLD_RANGE_HPP

namespace mold
{

Range::Range (size_t lower, size_t upper)
{
	if (lower > upper)
	{
		std::swap(lower, upper);
	}
	lower_ = lower;
	upper_ = upper;
}

clay::Shape Range::apply (const clay::Shape& inshape) const
{
	size_t n = inshape.rank();
	if (lower_ >= n)
	{
		return clay::Shape();
	}
	size_t upper = std::min(n, upper_);
	auto bt = inshape.begin();
	if (lower_ == upper) return {};
	return std::vector<size_t>(bt + lower_, bt + upper);
}

clay::Shape Range::split (const clay::Shape& inshape) const
{
	size_t n = inshape.rank();
	if (lower_ >= n)
	{
		return inshape;
	}
	size_t upper = std::min(n, upper_);
	auto bt = inshape.begin();
	if (lower_ == upper) return inshape;
	std::vector<size_t> out(bt, bt + lower_);
	out.insert(out.end(), bt + upper, inshape.end());
	return out;
}

}

#endif /* MOLD_RANGE_HPP */
