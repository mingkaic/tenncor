//
//  registry.cpp
//  slip
//

#include <algorithm>

#include "slip/include/operations.hpp"
#include "slip/registry.hpp"
#include "slip/error.hpp"

#include "clay/error.hpp"

#include "ioutil/stream.hpp"

#ifdef SLIP_REGISTRY_HPP

namespace slip
{

using ArgsF = std::function<void(clay::State&,std::vector<clay::State>)>;

using ShaperF = std::function<clay::Shape(std::vector<clay::State>)>;

using TyperF = std::function<clay::DTYPE(std::vector<clay::DTYPE>)>;

using TypeReg = std::unordered_map<clay::DTYPE,ArgsF,EnumHash>;

#define REGISTER_FUNC(CODE, FUNC) {\
slip::CODE, TypeReg{\
{ clay::DOUBLE, slip::FUNC<double> },{ clay::FLOAT, slip::FUNC<float> },\
{ clay::INT8, slip::FUNC<int8_t> },{ clay::UINT8, slip::FUNC<uint8_t> },\
{ clay::INT16, slip::FUNC<int16_t> },{ clay::UINT16, slip::FUNC<uint16_t> },\
{ clay::INT32, slip::FUNC<int32_t> },{ clay::UINT32, slip::FUNC<uint32_t> },\
{ clay::INT64, slip::FUNC<int64_t> },{ clay::UINT64, slip::FUNC<uint64_t> } } },

#define REGISTER_SFUNC(CODE, FUNC) {\
slip::CODE, TypeReg{\
{ clay::DOUBLE, slip::FUNC },{ clay::FLOAT, slip::FUNC },\
{ clay::INT8, slip::FUNC },{ clay::UINT8, slip::FUNC },\
{ clay::INT16, slip::FUNC },{ clay::UINT16, slip::FUNC },\
{ clay::INT32, slip::FUNC },{ clay::UINT32, slip::FUNC },\
{ clay::INT64, slip::FUNC },{ clay::UINT64, slip::FUNC } } },

static std::unordered_map<OPCODE,TypeReg,EnumHash> op_registry =
{
	REGISTER_FUNC(CAST, cast)

	REGISTER_FUNC(ABS, abs)
	REGISTER_FUNC(NEG, neg)
	REGISTER_FUNC(NOT, logic_not)
	REGISTER_FUNC(SIN, sin)
	REGISTER_FUNC(COS, cos)
	REGISTER_FUNC(TAN, tan)
	REGISTER_FUNC(EXP, exp)
	REGISTER_FUNC(LOG, log)
	REGISTER_FUNC(SQRT, sqrt)
	REGISTER_FUNC(ROUND, round)

	REGISTER_FUNC(POW, pow)
	REGISTER_FUNC(ADD, add)
	REGISTER_FUNC(SUB, sub)
	REGISTER_FUNC(MUL, mul)
	REGISTER_FUNC(DIV, div)
	REGISTER_FUNC(EQ, eq)
	REGISTER_FUNC(NE, neq)
	REGISTER_FUNC(LT, lt)
	REGISTER_FUNC(GT, gt)
	REGISTER_FUNC(BINO, rand_binom)
	REGISTER_FUNC(UNIF, rand_uniform)
	REGISTER_FUNC(NORM, rand_normal)

	REGISTER_FUNC(TRANSPOSE, transpose)
	REGISTER_FUNC(FLIP, flip)
	REGISTER_FUNC(EXPAND, expand)

	REGISTER_FUNC(UARGMAX, unar_argmax)
	REGISTER_FUNC(URMAX, unar_max)
	REGISTER_FUNC(URSUM, unar_sum)

	REGISTER_FUNC(ARGMAX, argmax)
	REGISTER_FUNC(RMAX, max)
	REGISTER_FUNC(RSUM, sum)

	REGISTER_SFUNC(N_ELEMS, n_elems)
	REGISTER_SFUNC(N_DIMS, n_dims)

	REGISTER_FUNC(MATMUL, matmul)
	REGISTER_FUNC(RESHAPE, copyover)
	REGISTER_FUNC(JACOBIAN, jacobian)
	REGISTER_FUNC(TRACE_EXPAND, trace_expand)
};

// proxy for either init or uninit operate
class OperateIO final : public mold::iOperateIO
{
public:
	OperateIO (OPCODE opcode, ShaperF shaper, TyperF typer) :
		shaper_(shaper), typer_(typer)
	{
		auto types = op_registry.find(opcode);
		if (op_registry.end() == types)
		{
			throw UnsupportedOpcodeError(opcode);
		}
		ops_ = types->second;
	}

	bool write_data (clay::State& dest,
		std::vector<clay::State> args) const override
	{
		auto imms = get_imms(args);
		bool success = dest.shape_.
			is_compatible_with(imms.first) &&
			dest.dtype_ == imms.second;
		if (success)
		{
			auto op = ops_.find(imms.second);
			if (ops_.end() == op)
			{
				throw clay::UnsupportedTypeError(imms.second);
			}
			op->second(dest, args);
		}
		return success;
	}

	mold::ImmPair get_imms (std::vector<clay::State> args) const override
	{
		if (args.empty())
		{
			throw NoArgumentsError();
		}
		std::vector<clay::DTYPE> types(args.size());
		std::transform(args.begin(), args.end(), types.begin(),
		[](clay::State& state) -> clay::DTYPE
		{
			return state.dtype_;
		});
		clay::DTYPE otype = typer_(types);
		return {shaper_(args), otype};
	}

private:
	iOperateIO* clone_impl (void) const override
	{
		return new OperateIO(*this);
	}

	ShaperF shaper_;

	TyperF typer_;

	TypeReg ops_;
};

struct OpWrapper
{
	ShaperF shaper_;
	TyperF typer_;
};

static clay::Shape elem_shape (std::vector<clay::State> states)
{
	if (states.empty())
	{
		throw NoArgumentsError();
	}
	clay::Shape out = states[0].shape_;
	for (auto it = states.begin() + 1, et = states.end();
		it != et; it++)
	{
		if (it->shape_.n_elems() != 1 &&
			false == it->shape_.is_compatible_with(out))
		{
			throw ShapeMismatchError(out, it->shape_);
		}
	}
	return out;
}

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

static clay::Shape scalar_shape (std::vector<clay::State> states)
{
	if (states.size() != 1)
	{
		throw BadNArgsError(1, states.size());
	}
	return clay::Shape({1});
}

static clay::DTYPE scalar_type (std::vector<clay::DTYPE> types)
{
	if (types.size() != 1)
	{
		throw BadNArgsError(1, types.size());
	}
	return types[0];
}

static clay::Shape reduce_shape (std::vector<clay::State> states)
{
	if (states.size() != 2)
	{
		throw BadNArgsError(2, states.size());
	}
	clay::State& state = states[1];
	if (1 != state.shape_.n_elems())
	{
		throw ShapeMismatchError(clay::Shape({1}), state.shape_);
	}
	uint64_t dim = *(safe_get<uint64_t>(state.data_));
	clay::Shape& shape = states[0].shape_;
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

static clay::Shape matmul_shape (std::vector<clay::State> states)
{
	if (states.size() != 2)
	{
		throw BadNArgsError(2, states.size());
	}
	clay::Shape& t1s = states[0].shape_;
	clay::Shape& t2s = states[1].shape_;

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

static std::unordered_map<OPCODE,OpWrapper,EnumHash> registry =
[]()
{
	OpWrapper elem{elem_shape, same_type};
	OpWrapper scalar{scalar_shape, scalar_type};
	OpWrapper reduce{reduce_shape, reduce_type};

	return std::unordered_map<OPCODE,OpWrapper,EnumHash>{
	{CAST, OpWrapper{
	[](std::vector<clay::State> states) -> clay::Shape
	{
		if (states.size() != 2)
		{
			throw BadNArgsError(2, states.size());
		}
		return states[1].shape_;
	},
	[](std::vector<clay::DTYPE> types) -> clay::DTYPE
	{
		if (types.size() != 2)
		{
			throw BadNArgsError(2, types.size());
		}
		return types[0];
	}}},
	{ABS, elem},
	{NEG, elem},
	{NOT, elem},
	{SIN, elem},
	{COS, elem},
	{TAN, elem},
	{EXP, elem},
	{LOG, elem},
	{SQRT, elem},
	{ROUND, elem},
	{POW, elem},
	{ADD, elem},
	{SUB, elem},
	{MUL, elem},
	{DIV, elem},
	{EQ, elem},
	{NE, elem},
	{GT, elem},
	{LT, elem},
	{BINO, OpWrapper{elem_shape,
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
	}}},
	{UNIF, elem},
	{NORM, OpWrapper{elem_shape,
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
	}}},
	{TRANSPOSE, OpWrapper{
	[](std::vector<clay::State> states) -> clay::Shape
	{
		if (states.empty())
		{
			throw NoArgumentsError();
		}
		clay::State state = states.front();
		std::vector<size_t> outlist = state.shape_.as_list();
		if (states.size() > 1)
		{
			clay::State& pstate = states[1];
			if (pstate.dtype_ != clay::UINT64)
			{
				throw clay::UnsupportedTypeError(pstate.dtype_);
			}
			uint64_t* ptr = safe_get<uint64_t>(pstate.data_);
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
	}}},
	{FLIP, OpWrapper{
	[](std::vector<clay::State> states) -> clay::Shape
	{
		if (2 != states.size())
		{
			throw BadNArgsError(2, states.size());
		}
		size_t rank = states[0].shape_.rank();
		clay::State& dstate = states[1];
		size_t ndims = dstate.shape_.n_elems();
		uint64_t* dims = safe_get<uint64_t>(dstate.data_);
		for (size_t i = 0; i < ndims; ++i)
		{
			if (dims[i] >= rank)
			{
				throw InvalidDimensionError(
					dims[i], states[0].shape_);
			}
		}
		return states.front().shape_;
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
	}}},
	{UARGMAX, scalar},
	{URMAX, scalar},
	{URSUM, scalar},
	{ARGMAX, reduce},
	{RMAX, reduce},
	{RSUM, reduce},
	{EXPAND, OpWrapper{
	[](std::vector<clay::State> states) -> clay::Shape
	{
		if (3 != states.size())
		{
			throw BadNArgsError(3, states.size());
		}
		if (1 != states[1].shape_.n_elems())
		{
			throw ShapeMismatchError(
				clay::Shape({1}),
				states[1].shape_);
		}
		if (1 != states[2].shape_.n_elems())
		{
			throw ShapeMismatchError(
				clay::Shape({1}),
				states[2].shape_);
		}
		clay::State& nstate = states[1];
		clay::State& dstate = states[2];
		uint64_t mul = *(safe_get<uint64_t>(nstate.data_));
		uint64_t dim = *(safe_get<uint64_t>(dstate.data_));
		std::vector<size_t> slist = states[0].shape_.as_list();
		if (slist.size() < dim)
		{
			throw InvalidDimensionError(dim, states[0].shape_);
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
	}}},
	{N_ELEMS, OpWrapper{
	[](std::vector<clay::State> states) -> clay::Shape
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
	}}},
	{N_DIMS, OpWrapper{
	[](std::vector<clay::State> states) -> clay::Shape
	{
		if (2 != states.size())
		{
			throw BadNArgsError(2, states.size());
		}
		clay::State& dstate = states[1];
		if (1 != dstate.shape_.n_elems())
		{
			throw ShapeMismatchError(
				clay::Shape({1}),
				dstate.shape_);
		}
		uint64_t dim = *(safe_get<uint64_t>(dstate.data_));
		if (dim >= states[0].shape_.rank())
		{
			throw InvalidDimensionError(dim, states[0].shape_);
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
	}}},
	{MATMUL, OpWrapper{matmul_shape, same_type}},
	{RESHAPE, OpWrapper{
	[](std::vector<clay::State> states) -> clay::Shape
	{
		if (2 != states.size())
		{
			throw BadNArgsError(2, states.size());
		}
		clay::State& shapes = states[1];
		clay::Shape& srcshape = states[0].shape_;
		uint64_t* dim = safe_get<uint64_t>(shapes.data_);
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
	}}},
	{JACOBIAN, OpWrapper{
	[](std::vector<clay::State> states) -> clay::Shape
	{
		if (3 != states.size())
		{
			throw BadNArgsError(3, states.size());
		}
		clay::State& dims = states[2];
		if (dims.shape_.n_elems() != 2)
		{
			throw std::runtime_error("failed to specify target and swap dimensions in jacobian");
		}
		uint64_t* dim = safe_get<uint64_t>(dims.data_);
		clay::Shape ashape = states[0].shape_;
		clay::Shape bshape = states[1].shape_;
		clay::Shape yshape = states[*dim].shape_;
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
	}}},
	{TRACE_EXPAND, OpWrapper{
	[](std::vector<clay::State> states) -> clay::Shape
	{
		if (2 != states.size())
		{
			throw BadNArgsError(2, states.size());
		}
		if (1 != states[1].shape_.n_elems())
		{
			throw ShapeMismatchError(
				clay::Shape({1}),
				states[1].shape_);
		}
		clay::State& dstate = states[1];
		uint64_t dim = *(safe_get<uint64_t>(dstate.data_));
		std::vector<size_t> slist = states[0].shape_.as_list();
		if (slist.size() <= dim)
		{
			throw InvalidDimensionError(dim, states[0].shape_);
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
	}}}};
}();

bool has_op (OPCODE opcode)
{
	return registry.end() != registry.find(opcode);
}

mold::iOperatePtrT get_op (OPCODE opcode)
{
	auto it = registry.find(opcode);
	if (registry.end() == it)
	{
		throw UnsupportedOpcodeError(opcode);
	}
	return mold::iOperatePtrT(new OperateIO(opcode,
		it->second.shaper_, it->second.typer_));
}

}

#endif
