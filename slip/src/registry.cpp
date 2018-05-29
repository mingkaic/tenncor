//
//  registry.cpp
//  slip
//

#include <algorithm>

#include "ioutil/stream.hpp"

#include "slip/include/operations.hpp"
#include "slip/registry.hpp"

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

	REGISTER_FUNC(ARGMAX, argmax)
	REGISTER_FUNC(RMAX, max)
	REGISTER_FUNC(RSUM, sum)

	REGISTER_SFUNC(N_ELEMS, n_elems)
	REGISTER_SFUNC(N_DIMS, n_dims)

	REGISTER_FUNC(MATMUL, matmul)
};

// proxy for either init or uninit operate
class OperateIO final : public mold::iOperateIO
{
public:
	OperateIO (OPCODE opcode, ShaperF shaper, TyperF typer) :
		opcode_(opcode), state_(new UninitOperate(shaper, typer)) {}

	OperateIO (const OperateIO& other) :
		state_(other.state_->clone()) {}

	OperateIO (OperateIO&&) = default;

	OperateIO& operator = (const OperateIO& other)
	{
		if (this != &other)
		{
			state_ = mold::iOperatePtrT(other.state_->clone());
		}
		return *this;
	}

	OperateIO& operator = (OperateIO&&) = default;

	bool read_data (clay::State& dest) const override
	{
		return state_->read_data(dest);
	}

	mold::ImmPair get_imms (void) override
	{
		return state_->get_imms();
	}

	void set_args (std::vector<clay::State> args) override
	{
		state_->set_args(args);
		if (UninitOperate* uop = static_cast<UninitOperate*>(state_.get()))
		{
			state_ = mold::iOperatePtrT(
				new InitOperate(uop->args_, uop->get_imms(), opcode_));
		}
	}

private:
	class UninitOperate final : public mold::iOperateIO
	{
	public:
		UninitOperate (ShaperF shaper, TyperF typer) :
			shaper_(shaper), typer_(typer) {}

		bool read_data (clay::State&) const override
		{
			throw std::bad_function_call(); // todo: add context
		}

		mold::ImmPair get_imms (void) override
		{
			if (args_.empty())
			{
				throw std::exception(); // todo: add context
			}
			std::vector<clay::DTYPE> types(args_.size());
			std::transform(args_.begin(), args_.end(), types.begin(),
			[](clay::State& state) -> clay::DTYPE
			{
				return state.dtype_;
			});
			clay::DTYPE otype = typer_(types);
			return {shaper_(args_), otype};
		}

		void set_args (std::vector<clay::State> args) override
		{
			args_ = args;
		}

		std::vector<clay::State> args_;

	private:
		iOperateIO* clone_impl (void) const override
		{
			return new UninitOperate(*this);
		}

		ShaperF shaper_;

		TyperF typer_;
	};

	class InitOperate final : public mold::iOperateIO
	{
	public:
		InitOperate (std::vector<clay::State> args,
			mold::ImmPair imms, OPCODE opcode) :
		args_(args), imms_(imms)
		{
			auto types = op_registry.find(opcode);
			if (op_registry.end() == types)
			{
				throw std::exception(); // todo: add context
			}
			auto it = types->second.find(imms.second);
			if (types->second.end() == it)
			{
				throw std::exception(); // todo: add context
			}
			op_ = it->second;
		}

		bool read_data (clay::State& dest) const override
		{
			bool success = dest.shape_.
				is_compatible_with(imms_.first) &&
				dest.dtype_ == imms_.second;
			if (success)
			{
				op_(dest, args_);
			}
			return success;
		}

		mold::ImmPair get_imms (void) override
		{
			return imms_;
		}

		void set_args (std::vector<clay::State> args) override
		{
			throw std::bad_function_call(); // todo: add context
		}

	private:
		iOperateIO* clone_impl (void) const override
		{
			return new InitOperate(*this);
		}

		std::vector<clay::State> args_;

		mold::ImmPair imms_;

		ArgsF op_;
	};

	iOperateIO* clone_impl (void) const override
	{
		return new OperateIO(*this);
	}

	OPCODE opcode_;

	mold::iOperatePtrT state_;
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
		throw std::exception(); // todo: add context
	}
	clay::Shape out = states[0].shape_;
	if (false == std::all_of(states.begin() + 1, states.end(),
	[&out](clay::State& state)
	{
		return state.shape_.n_elems() == 1 || state.shape_.is_compatible_with(out);
	}))
	{
		throw std::exception(); // todo: add context
	}
	return out;
}

static clay::DTYPE same_type (std::vector<clay::DTYPE> types)
{
	if (types.empty())
	{
		throw std::exception(); // todo: add context
	}
	clay::DTYPE out = types[0];
	if (false == std::all_of(types.begin() + 1, types.end(),
	[&out](clay::DTYPE& dtype)
	{
		return dtype == out;
	}))
	{
		throw std::exception(); // todo: add context
	}
	return out;
}

static clay::Shape reduce_shape (std::vector<clay::State> states)
{
	if (states.empty())
	{
		throw std::exception(); // todo: add context
	}
	clay::Shape out;
	if (states.size() > 1)
	{
		clay::State& state = states[1];
		if (1 != state.shape_.n_elems())
		{
			throw std::exception();
		}
		uint64_t dim = *(safe_get<uint64_t>(state.data_));
		clay::Shape& shape = states[0].shape_;
		if (dim >= shape.rank())
		{
			throw std::exception();
		}
		out = {shape[dim]};
	}
	else
	{
		out = {1};
	}
	return out;
}

static clay::DTYPE reduce_type (std::vector<clay::DTYPE> types)
{
	if (types.empty())
	{
		throw std::exception(); // todo: add context
	}
	if (types.size() > 1 && clay::UINT64 != types[1])
	{
		throw std::exception();
	}
	return types[0];
}

static clay::Shape matmul_shape (std::vector<clay::State> states)
{
	if (states.size() != 2)
	{
		throw std::exception(); // todo: add context
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
	OpWrapper reduce{reduce_shape, reduce_type};

	return std::unordered_map<OPCODE,OpWrapper,EnumHash>{
	{CAST, OpWrapper{
	[](std::vector<clay::State> states) -> clay::Shape
	{
		if (states.size() != 2)
		{
			throw std::exception(); // todo: add context
		}
		return states[1].shape_;
	},
	[](std::vector<clay::DTYPE> types) -> clay::DTYPE
	{
		if (types.size() != 2)
		{
			throw std::exception(); // todo: add context
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
			throw std::exception(); // todo: add context
		}
		if (types[0] == clay::DOUBLE ||
			types[0] == clay::FLOAT)
		{
			throw std::exception(); // todo: add context
		}
		if (types[1] != clay::DOUBLE)
		{
			throw std::exception(); // todo: add context
		}
		return types[0];
	}}},
	{UNIF, elem},
	{NORM, OpWrapper{elem_shape,
	[](std::vector<clay::DTYPE> types) -> clay::DTYPE
	{
		if (types.size() != 2)
		{
			throw std::exception(); // todo: add context
		}
		if ((types[0] != clay::DOUBLE && types[0] != clay::FLOAT) ||
			types[0] != types[1])
		{
			throw std::exception(); // todo: add context
		}
		return types[0];
	}}},
	{TRANSPOSE, OpWrapper{
	[](std::vector<clay::State> states) -> clay::Shape
	{
		if (states.empty())
		{
			throw std::exception(); // todo: add context
		}
		clay::State state = states.front();
		std::vector<size_t> outlist = state.shape_.as_list();
		if (states.size() > 1)
		{
			clay::State& pstate = states[1];
			if (pstate.dtype_ != clay::UINT64)
			{
				throw std::exception();
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
			throw std::exception(); // todo: add context
		}
		if (types.size() > 1 && types[1] != clay::UINT64)
		{
			throw std::exception(); // todo: add context
		}
		return types[0];
	}}},
	{FLIP, OpWrapper{
	[](std::vector<clay::State> states) -> clay::Shape
	{
		if (2 != states.size())
		{
			throw std::exception(); // todo: add context
		}
		size_t rank = states[0].shape_.rank();
		clay::State& dstate = states[1];
		size_t ndims = dstate.shape_.n_elems();
		uint64_t* dims = safe_get<uint64_t>(dstate.data_);
		if (std::any_of(dims, dims + ndims,
		[rank](uint64_t d)
		{
			return d >= rank;
		}))
		{
			throw std::exception();
		}
		return states.front().shape_;
	},
	[](std::vector<clay::DTYPE> types) -> clay::DTYPE
	{
		if (2 != types.size())
		{
			throw std::exception(); // todo: add context
		}
		if (types[1] != clay::UINT64)
		{
			throw std::exception(); // todo: add context
		}
		return types[0];
	}}},
	{ARGMAX, reduce},
	{RMAX, reduce},
	{RSUM, reduce},
	{EXPAND, OpWrapper{
	[](std::vector<clay::State> states) -> clay::Shape
	{
		if (3 != states.size())
		{
			throw std::exception(); // todo: add context
		}
		if (1 != states[1].shape_.n_elems() ||
			1 != states[2].shape_.n_elems())
		{
			throw std::exception(); // todo: add context
		}
		clay::State& nstate = states[1];
		clay::State& dstate = states[2];
		uint64_t mul = *(safe_get<uint64_t>(nstate.data_));
		uint64_t dim = *(safe_get<uint64_t>(dstate.data_));
		std::vector<size_t> slist = states[0].shape_.as_list();
		if (slist.size() < dim)
		{
			throw std::exception();
		}
		slist.insert(slist.begin() + dim, mul);
		return clay::Shape(slist);
	},
	[](std::vector<clay::DTYPE> types) -> clay::DTYPE
	{
		if (3 != types.size())
		{
			throw std::exception(); // todo: add context
		}
		if (types[1] != clay::UINT64 ||
			types[2] != clay::UINT64)
		{
			throw std::exception(); // todo: add context
		}
		return types[0];
	}}},
	{N_ELEMS, OpWrapper{
	[](std::vector<clay::State> states) -> clay::Shape
	{
		if (states.empty())
		{
			throw std::exception(); // todo: add context
		}
		return clay::Shape(std::vector<size_t>{1});
	},
	[](std::vector<clay::DTYPE> types) -> clay::DTYPE
	{
		if (types.empty())
		{
			throw std::exception(); // todo: add context
		}
		return clay::UINT64;
	}}},
	{N_DIMS, OpWrapper{
	[](std::vector<clay::State> states) -> clay::Shape
	{
		if (2 != states.size())
		{
			throw std::exception(); // todo: add context
		}
		clay::State& dstate = states[1];
		if (1 != dstate.shape_.n_elems())
		{
			throw std::exception(); // todo: add context
		}
		uint64_t dim = *(safe_get<uint64_t>(dstate.data_));
		if (dim >= states[0].shape_.rank())
		{
			throw std::exception();
		}
		return clay::Shape(std::vector<size_t>{1});
	},
	[](std::vector<clay::DTYPE> types) -> clay::DTYPE
	{
		if (2 != types.size())
		{
			throw std::exception(); // todo: add context
		}
		return clay::UINT64;
	}}},
	{MATMUL, OpWrapper{matmul_shape, same_type}}};
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
		throw std::exception();
	}
	return mold::iOperatePtrT(new OperateIO(opcode,
		it->second.shaper_, it->second.typer_));
}

}

#endif
