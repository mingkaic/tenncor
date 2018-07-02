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

clay::Shape scalar_shape (std::vector<mold::StateRange> states)
{
	if (states.size() != 1)
	{
		throw BadNArgsError(1, states.size());
	}
	mold::StateRange& srange = states[0];
	clay::Shape outer = srange.outer();
	if (false == outer.is_part_defined())
	{
		return clay::Shape({1});
	}
	return outer;
}

clay::Shape reduce_shape (std::vector<mold::StateRange> states)
{
	if (states.size() != 2)
	{
		throw BadNArgsError(2, states.size());
	}
	clay::State& state = states[1].arg_;
	if (1 != state.shape_.n_elems())
	{
		throw ShapeMismatchError(clay::Shape({1}), state.shape_);
	}
	uint64_t dim = *(safe_get<uint64_t>(state));
	clay::Shape shape = states[0].shape();
	if (dim >= shape.rank())
	{
		throw InvalidDimensionError(dim, shape);
	}
	std::vector<size_t> slist = shape.as_list();
	if (1 == slist.size())
	{
		slist[0] = 1;
	}
	else
	{
		slist.erase(slist.begin() + dim);
	}
	return clay::Shape(slist);
}

clay::Shape matmul_shape (std::vector<mold::StateRange> states)
{
	if (states.size() != 2)
	{
		throw BadNArgsError(2, states.size());
	}
	clay::Shape t1s = states[0].shape();
	clay::Shape t2s = states[1].shape();

	std::vector<size_t> al = t1s.as_list();
	std::vector<size_t> bl = t2s.as_list();
	size_t rank1 = t1s.rank();
	size_t rank2 = t2s.rank();

	// account for vectors
	size_t ax = rank1 ? al[0] : 0;
	size_t ay = rank1> 1 ? al[1] : 1;
	size_t bx = rank2 ? bl[0] : 0;
	size_t by = rank2> 1 ? bl[1] : 1;

	// ensure the dimensions beyond 2d are equal
	size_t minend = std::min(rank1, rank2);
	std::vector<size_t> beyond2d;
	if (minend> 2)
	{
		auto ait = al.begin()+2;
		auto aet = al.begin()+minend;
		if (std::equal(ait, aet, bl.begin()+2))
		{
			beyond2d.insert(beyond2d.end(), ait, aet);
		}
		else
		{
			ioutil::Stream s;
			s << "attempting to matrix multiple shapes "
				<< t1s.as_list() << " and " << t2s.as_list();
			throw std::logic_error(s.str());
		}
		// check that remaining shape values are ones,
		// otherwise one shape is larger than the other
		auto it = rank1> rank2 ? al.begin() : bl.begin();
		auto et = rank1> rank2 ? al.end() : bl.end();
		if (!std::all_of(it + minend, et, [](size_t e) { return e == 1; }))
		{
			ioutil::Stream s;
			s << "attempting to matrix multiple different shapes "
				<< t1s.as_list() << " and " << t2s.as_list();
			throw std::logic_error(s.str());
		}
	}

	// get resulting shape
	std::vector<size_t> res_shape;
	if (ax == by)
	{
		res_shape = {bx, ay};
	}
	else
	{
		ioutil::Stream s;
		s << "matmul shapes " << t1s.as_list()
			<< " and " << t2s.as_list() << " do not match";
		throw std::logic_error(s.str());
	}
	res_shape.insert(res_shape.end(), beyond2d.begin(), beyond2d.end());
	return clay::Shape(res_shape);
}

}

#endif /* SLIP_SHAPER_HPP */
