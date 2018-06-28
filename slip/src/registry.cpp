//
//  registry.cpp
//  slip
//

#include <algorithm>

#include "slip/include/operations.hpp"
#include "slip/include/operate_io.hpp"
#include "slip/registry.hpp"

#include "ioutil/stream.hpp"

#ifdef SLIP_REGISTRY_HPP

namespace slip
{

#define TMAP_FUNC(FUNC) TypeRegT{\
{ clay::DOUBLE, FUNC<double> },{ clay::FLOAT, FUNC<float> },\
{ clay::INT8, FUNC<int8_t> },{ clay::UINT8, FUNC<uint8_t> },\
{ clay::INT16, FUNC<int16_t> },{ clay::UINT16, FUNC<uint16_t> },\
{ clay::INT32, FUNC<int32_t> },{ clay::UINT32, FUNC<uint32_t> },\
{ clay::INT64, FUNC<int64_t> },{ clay::UINT64, FUNC<uint64_t> } }

#define TMAP_SFUNC(FUNC) TypeRegT{\
{ clay::DOUBLE, FUNC },{ clay::FLOAT, FUNC },\
{ clay::INT8, FUNC },{ clay::UINT8, FUNC },\
{ clay::INT16, FUNC },{ clay::UINT16, FUNC },\
{ clay::INT32, FUNC },{ clay::UINT32, FUNC },\
{ clay::INT64, FUNC },{ clay::UINT64, FUNC } }

// SHAPE HANDLERS

static clay::Shape elem_shape (std::vector<mold::StateRange> states)
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
		if (out.n_elems() == 1)
		{
			out = oshape;
			outshape = it->shape();
		}
		else if (oshape.n_elems() != 1 &&
			false == oshape.is_compatible_with(out))
		{
			throw ShapeMismatchError(out, oshape);
		}
	}
	return outshape;
}

static clay::Shape scalar_shape (std::vector<mold::StateRange> states)
{
	if (states.size() != 1)
	{
		throw BadNArgsError(1, states.size());
	}
	return clay::Shape({1});
}

static clay::Shape reduce_shape (std::vector<mold::StateRange> states)
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

static clay::Shape matmul_shape (std::vector<mold::StateRange> states)
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

// TYPE HANDLERS

static clay::DTYPE same_type (std::vector<clay::DTYPE> types)
{
	if (types.empty())
	{
		throw NoArgumentsError();
	}
	clay::DTYPE out = types[0];
	for (auto it = types.begin() + 1, et = types.end();
		it != et; it++)
	{
		if (*it != out)
		{
			throw TypeMismatchError(out, *it);
		}
	}
	return out;
}

static clay::DTYPE reduce_type (std::vector<clay::DTYPE> types)
{
	if (types.size() != 2)
	{
		throw BadNArgsError(2, types.size());
	}
	if (clay::UINT64 != types[1])
	{
		throw clay::UnsupportedTypeError(types[1]);
	}
	return types[0];
}

// REGISTRY DEFINITION

#define MAKE_OP(treg, shaper, typer)\
mold::OperatePtrT(new OperateIO(treg, shaper, typer))
#define ELEM(op) MAKE_OP(TMAP_FUNC(op), elem_shape, same_type)
#define SCALAR(op) MAKE_OP(TMAP_FUNC(op), scalar_shape, same_type)
#define REDUCE(op) MAKE_OP(TMAP_FUNC(op), reduce_shape, reduce_type)

static EnumMap<OPCODE,mold::OperatePtrT> registry =
{
	{CAST, MAKE_OP(TMAP_FUNC(cast),
	[](std::vector<mold::StateRange> states) -> clay::Shape
	{
		if (states.size() != 2)
		{
			throw BadNArgsError(2, states.size());
		}
		return states[1].shape();
	},
	[](std::vector<clay::DTYPE> types) -> clay::DTYPE
	{
		if (types.size() != 2)
		{
			throw BadNArgsError(2, types.size());
		}
		return types[0];
	})},
	{ABS, ELEM(abs)},
	{NEG, ELEM(neg)},
	{NOT, ELEM(logic_not)},
	{SIN, ELEM(sin)},
	{COS, ELEM(cos)},
	{TAN, ELEM(tan)},
	{EXP, ELEM(exp)},
	{LOG, ELEM(log)},
	{SQRT, ELEM(sqrt)},
	{ROUND, ELEM(round)},
	{POW, ELEM(pow)},
	{ADD, ELEM(add)},
	{SUB, ELEM(sub)},
	{MUL, ELEM(mul)},
	{DIV, ELEM(div)},
	{EQ, ELEM(eq)},
	{NE, ELEM(neq)},
	{GT, ELEM(gt)},
	{LT, ELEM(lt)},
	{BINO, MAKE_OP(TMAP_FUNC(rand_binom), elem_shape,
	[](std::vector<clay::DTYPE> types) -> clay::DTYPE
	{
		if (types.size() != 2)
		{
			throw BadNArgsError(2, types.size());
		}
		if (types[0] == clay::DOUBLE ||
			types[0] == clay::FLOAT)
		{
			throw clay::UnsupportedTypeError(types[0]);
		}
		if (types[1] != clay::DOUBLE)
		{
			throw clay::UnsupportedTypeError(types[1]);
		}
		return types[0];
	})},
	{UNIF, ELEM(rand_uniform)},
	{NORM, MAKE_OP(TMAP_FUNC(rand_normal), elem_shape,
	[](std::vector<clay::DTYPE> types) -> clay::DTYPE
	{
		if (types.size() != 2)
		{
			throw BadNArgsError(2, types.size());
		}
		if (types[0] != clay::DOUBLE && types[0] != clay::FLOAT)
		{
			throw clay::UnsupportedTypeError(types[0]);
		}
		if (types[0] != types[1])
		{
			throw TypeMismatchError(types[0], types[1]);
		}
		return types[0];
	})},
	{TRANSPOSE, MAKE_OP(TMAP_FUNC(transpose),
	[](std::vector<mold::StateRange> states) -> clay::Shape
	{
		if (states.empty())
		{
			throw NoArgumentsError();
		}
		clay::State& state = states.front().arg_;
		std::vector<size_t> outlist = state.shape_.as_list();
		if (states.size() > 1)
		{
			clay::State& pstate = states[1].arg_;
			if (pstate.dtype_ != clay::UINT64)
			{
				throw clay::UnsupportedTypeError(pstate.dtype_);
			}
			uint64_t* ptr = safe_get<uint64_t>(pstate);
			std::vector<size_t> perm(ptr, ptr + pstate.shape_.n_elems());
			std::vector<size_t> inlist = outlist;
			for (size_t i = 0; i < perm.size(); ++i)
			{
				if (i != perm[i])
				{
					outlist[i] = inlist[perm[i]];
				}
			}
		}
		else
		{
			std::reverse(outlist.begin(), outlist.end());
		}
		return clay::Shape(outlist);
	},
	[](std::vector<clay::DTYPE> types) -> clay::DTYPE
	{
		if (types.empty())
		{
			throw NoArgumentsError();
		}
		if (types.size() > 1 && types[1] != clay::UINT64)
		{
			throw clay::UnsupportedTypeError(types[1]);
		}
		return types[0];
	})},
	{FLIP, MAKE_OP(TMAP_FUNC(flip),
	[](std::vector<mold::StateRange> states) -> clay::Shape
	{
		if (2 != states.size())
		{
			throw BadNArgsError(2, states.size());
		}
		size_t rank = states[0].shape().rank();
		clay::State& dstate = states[1].arg_;
		size_t ndims = dstate.shape_.n_elems();
		uint64_t* dims = safe_get<uint64_t>(dstate);
		for (size_t i = 0; i < ndims; ++i)
		{
			if (dims[i] >= rank)
			{
				throw InvalidDimensionError(
					dims[i], states[0].shape());
			}
		}
		return states.front().shape();
	},
	[](std::vector<clay::DTYPE> types) -> clay::DTYPE
	{
		if (2 != types.size())
		{
			throw BadNArgsError(2, types.size());
		}
		if (types[1] != clay::UINT64)
		{
			throw clay::UnsupportedTypeError(types[1]);
		}
		return types[0];
	})},
	{UARGMAX, SCALAR(unar_argmax)},
	{URMAX, SCALAR(unar_max)},
	{URSUM, SCALAR(unar_sum)},
	{ARGMAX, REDUCE(argmax)},
	{RMAX, REDUCE(max)},
	{RSUM, REDUCE(sum)},
	{EXPAND, MAKE_OP(TMAP_FUNC(expand),
	[](std::vector<mold::StateRange> states) -> clay::Shape
	{
		if (3 != states.size())
		{
			throw BadNArgsError(3, states.size());
		}
		if (1 != states[1].shape().n_elems())
		{
			throw ShapeMismatchError(
				clay::Shape({1}),
				states[1].shape());
		}
		if (1 != states[2].shape().n_elems())
		{
			throw ShapeMismatchError(
				clay::Shape({1}),
				states[2].shape());
		}
		clay::State& nstate = states[1].arg_;
		clay::State& dstate = states[2].arg_;
		uint64_t mul = *(safe_get<uint64_t>(nstate));
		uint64_t dim = *(safe_get<uint64_t>(dstate));
		std::vector<size_t> slist = states[0].shape().as_list();
		if (slist.size() < dim)
		{
			throw InvalidDimensionError(dim, states[0].shape());
		}
		slist.insert(slist.begin() + dim, mul);
		return clay::Shape(slist);
	},
	[](std::vector<clay::DTYPE> types) -> clay::DTYPE
	{
		if (3 != types.size())
		{
			throw BadNArgsError(3, types.size());
		}
		if (types[1] != clay::UINT64)
		{
			throw clay::UnsupportedTypeError(types[1]);
		}
		if (types[2] != clay::UINT64)
		{
			throw clay::UnsupportedTypeError(types[2]);
		}
		return types[0];
	})},
	{N_ELEMS, MAKE_OP(TMAP_SFUNC(n_elems),
	[](std::vector<mold::StateRange> states) -> clay::Shape
	{
		if (states.empty())
		{
			throw NoArgumentsError();
		}
		return clay::Shape(std::vector<size_t>{1});
	},
	[](std::vector<clay::DTYPE> types) -> clay::DTYPE
	{
		if (types.empty())
		{
			throw NoArgumentsError();
		}
		return clay::UINT64;
	})},
	{N_DIMS, MAKE_OP(TMAP_SFUNC(n_dims),
	[](std::vector<mold::StateRange> states) -> clay::Shape
	{
		if (2 != states.size())
		{
			throw BadNArgsError(2, states.size());
		}
		clay::State& dstate = states[1].arg_;
		if (1 != dstate.shape_.n_elems())
		{
			throw ShapeMismatchError(
				clay::Shape({1}),
				dstate.shape_);
		}
		uint64_t dim = *(safe_get<uint64_t>(dstate));
		if (dim >= states[0].shape().rank())
		{
			throw InvalidDimensionError(dim, states[0].shape());
		}
		return clay::Shape(std::vector<size_t>{1});
	},
	[](std::vector<clay::DTYPE> types) -> clay::DTYPE
	{
		if (2 != types.size())
		{
			throw BadNArgsError(2, types.size());
		}
		// todo: add test for this, then uncomment
		// if (types[1] != clay::UINT64)
		// {
		// 	throw clay::UnsupportedTypeError(types[1]);
		// }
		return clay::UINT64;
	})},
	{MATMUL, MAKE_OP(TMAP_FUNC(matmul), matmul_shape, same_type)},
	{RESHAPE, MAKE_OP(TMAP_FUNC(copyover),
	[](std::vector<mold::StateRange> states) -> clay::Shape
	{
		if (2 != states.size())
		{
			throw BadNArgsError(2, states.size());
		}
		clay::State& shapes = states[1].arg_;
		clay::Shape srcshape = states[0].shape();
		uint64_t* dim = safe_get<uint64_t>(shapes);
		clay::Shape replshape(
			std::vector<size_t>{dim, dim + shapes.shape_.n_elems()});
		if (srcshape.n_elems() != replshape.n_elems())
		{
			throw std::exception(); // todo: add context
		}
		return replshape;
	},
	[](std::vector<clay::DTYPE> types) -> clay::DTYPE
	{
		if (2 != types.size())
		{
			throw BadNArgsError(2, types.size());
		}
		if (types[1] != clay::UINT64)
		{
			throw clay::UnsupportedTypeError(types[1]);
		}
		return types[0];
	})},
	{JACOBIAN, MAKE_OP(TMAP_FUNC(jacobian),
	[](std::vector<mold::StateRange> states) -> clay::Shape
	{
		if (3 != states.size())
		{
			throw BadNArgsError(3, states.size());
		}
		clay::State& dims = states[2].arg_;
		if (dims.shape_.n_elems() != 2)
		{
			throw std::runtime_error("failed to specify target and swap dimensions in jacobian");
		}
		uint64_t* dim = safe_get<uint64_t>(dims);
		clay::Shape ashape = states[0].shape();
		clay::Shape bshape = states[1].shape();
		clay::Shape yshape = states[*dim].shape();
		size_t x = bshape.at(0);
		size_t y = yshape.at(0);
		if (ashape.rank() > 1)
		{
			x *= ashape.at(1);
		}
		if (yshape.rank() > 1)
		{
			y *= yshape.at(1);
		}
		std::vector<size_t> slist = yshape.as_list();
		slist[0] = x;
		slist[1] = y;
		return clay::Shape(slist);
	},
	[](std::vector<clay::DTYPE> types) -> clay::DTYPE
	{
		if (3 != types.size())
		{
			throw BadNArgsError(3, types.size());
		}
		if (types[0] != types[1])
		{
			throw TypeMismatchError(types[0], types[1]);
		}
		if (types[2] != clay::UINT64)
		{
			throw clay::UnsupportedTypeError(types[2]);
		}
		return types[0];
	})},
	{TRACE_EXPAND, MAKE_OP(TMAP_FUNC(trace_expand),
	[](std::vector<mold::StateRange> states) -> clay::Shape
	{
		if (2 != states.size())
		{
			throw BadNArgsError(2, states.size());
		}
		if (1 != states[1].shape().n_elems())
		{
			throw ShapeMismatchError(
				clay::Shape({1}),
				states[1].shape());
		}
		clay::State& dstate = states[1].arg_;
		uint64_t dim = *(safe_get<uint64_t>(dstate));
		std::vector<size_t> slist = states[0].shape().as_list();
		if (slist.size() <= dim)
		{
			throw InvalidDimensionError(dim, states[0].shape());
		}
		slist.insert(slist.begin() + dim, slist[dim]);
		return clay::Shape(slist);
	},
	[](std::vector<clay::DTYPE> types) -> clay::DTYPE
	{
		if (2 != types.size())
		{
			throw BadNArgsError(3, types.size());
		}
		if (types[1] != clay::UINT64)
		{
			throw clay::UnsupportedTypeError(types[1]);
		}
		return types[0];
	})}
};

bool has_op (OPCODE opcode)
{
	return registry.end() != registry.find(opcode);
}

mold::OperatePtrT get_op (OPCODE opcode)
{
	auto it = registry.find(opcode);
	if (registry.end() == it)
	{
		throw UnsupportedOpcodeError(opcode);
	}
	return it->second;
}

}

#endif
