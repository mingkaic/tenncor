//
//  shaper.cpp
//  slip
//

#include "slip/include/shaper.hpp"
#include "slip/include/operations.hpp"
#include "slip/error.hpp"

#include "ioutil/stream.hpp"

#ifdef SLIP_SHAPER_HPP

namespace slip
{

clay::Shape elem_shape (std::vector<mold::StateRange> states)
{
	if (states.empty())
	{
		throw NoArgumentsError();
	}
	clay::Shape outshape = states.front().shape();
	clay::Shape in = states.front().inner();
	clay::Shape out = states.front().outer();
	for (auto it = states.begin() + 1, et = states.end();
		it != et; it++)
	{
		clay::Shape ishape = it->inner();
		clay::Shape oshape = it->outer();
		if (false == ishape.is_compatible_with(in))
		{
			throw ShapeMismatchError(in, ishape);
		}
		if (out.n_elems() != 1 && oshape.n_elems() != 1 &&
			false == oshape.is_compatible_with(out))
		{
			throw ShapeMismatchError(out, oshape);
		}
		if (it->shape().n_elems() > outshape.n_elems())
		{
			outshape = it->shape();
		}
	}
	return outshape;
}

clay::Shape relem_shape (std::vector<mold::StateRange> states)
{
	if (states.size() != 1)
	{
		throw BadNArgsError(1, states.size());
	}
	mold::StateRange& srange = states[0];
	clay::Shape inner = srange.inner();
	if (false == inner.is_fully_defined())
	{
		throw InvalidRangeError(srange.drange_, srange.shape());
	}
	return srange.shape();
}

clay::Shape reduce_shape (std::vector<mold::StateRange> states)
{
	if (states.size() != 1)
	{
		throw BadNArgsError(1, states.size());
	}
	mold::StateRange& srange = states[0];
	clay::Shape inner = srange.inner();
	if (false == inner.is_fully_defined())
	{
		throw InvalidRangeError(srange.drange_, srange.shape());
	}
	clay::Shape outer = srange.outer();
	if (false == outer.is_part_defined())
	{
		return clay::Shape({1});
	}
	return outer;
}

clay::Shape matmul_shape (std::vector<mold::StateRange> states)
{
	if (states.size() != 2)
	{
		throw BadNArgsError(2, states.size());
	}

	clay::Shape ins1 = states[0].inner();
	clay::Shape ins2 = states[1].inner();
	size_t rank1 = ins1.rank();
	size_t rank2 = ins2.rank();
	if (0 == rank1 || 2 > rank1)
	{
		throw InvalidRangeError(states[0].drange_, states[0].shape());
	}
	if (0 == rank2 || 2 > rank2)
	{
		throw InvalidRangeError(states[1].drange_, states[1].shape());
	}

	// ensure the dimensions beyond 2d are equal
	if (false == states[0].outer().is_compatible_with(states[1].outer()))
	{
		throw ShapeMismatchError(states[0].shape(), states[1].shape());
	}

	// account for vectors
	size_t ax = rank1 ? ins1.at(0) : 0;
	size_t ay = rank1 > 1 ? ins1.at(1) : 1;
	size_t bx = rank2 ? ins2.at(0) : 0;
	size_t by = rank2 > 1 ? ins2.at(1) : 1;

	// get resulting shape
	clay::Shape innershape;
	if (ax == by)
	{
		innershape = {bx, ay};
	}
	else
	{
		throw std::logic_error("matmul shapes " + clay::to_string(ins1) +
			" and " + clay::to_string(ins2) + " do not match");
	}
	clay::Shape front = concatenate(states[0].front(), innershape);
	return concatenate(front, states[0].back());
}

}

#endif /* SLIP_SHAPER_HPP */
