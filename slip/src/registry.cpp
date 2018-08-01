//
//  registry.cpp
//  slip
//

#include <algorithm>

#include "slip/include/operations.hpp"
#include "slip/include/operate_io.hpp"
#include "slip/include/shaper.hpp"
#include "slip/registry.hpp"

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

// REGISTRY DEFINITION

#define MAKE_OP(treg, shaper, typer)\
mold::OperatePtrT(new OperateIO(treg, shaper, typer))
#define ELEM(op) MAKE_OP(TMAP_FUNC(op), elem_shape, same_type)
#define RELEM(op) MAKE_OP(TMAP_FUNC(op), relem_shape, same_type)
#define REDUCE(op) MAKE_OP(TMAP_FUNC(op), reduce_shape, same_type)

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
	{ISMAX, RELEM(is_max)},
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
	{ARGMAX, REDUCE(arg_max)},
	{RMAX, REDUCE(rmax)},
	{RSUM, REDUCE(rsum)},
	{EXPAND, MAKE_OP(TMAP_FUNC(expand),
	[](std::vector<mold::StateRange> states) -> clay::Shape
	{
		if (2 != states.size())
		{
			throw BadNArgsError(3, states.size());
		}
		if (1 != states[1].shape().n_elems())
		{
			throw ShapeMismatchError(
				clay::Shape({1}),
				states[1].shape());
		}
		clay::Shape srcshape = states[0].shape();
		size_t dim = states[0].drange_.lower_;
		if (dim != states[0].drange_.upper_)
		{
			throw InvalidRangeError(states[0].drange_, srcshape);
		}
		clay::State& nstate = states[1].arg_;
		uint64_t mul = *(safe_get<uint64_t>(nstate));
		std::vector<size_t> slist = srcshape.as_list();
		if (slist.size() < dim)
		{
			throw InvalidDimensionError(dim, srcshape);
		}
		slist.insert(slist.begin() + dim, mul);
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
		if (1 != states.size())
		{
			throw BadNArgsError(1, states.size());
		}
		size_t dim = states[0].drange_.lower_;
		clay::Shape shape = states[0].shape();
		if (dim != states[0].drange_.upper_)
		{
			throw InvalidRangeError(states[0].drange_, shape);
		}
		if (dim >= shape.rank())
		{
			throw InvalidDimensionError(dim, shape);
		}
		return clay::Shape(std::vector<size_t>{1});
	},
	[](std::vector<clay::DTYPE> types) -> clay::DTYPE
	{
		if (1 != types.size())
		{
			throw BadNArgsError(1, types.size());
		}
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
