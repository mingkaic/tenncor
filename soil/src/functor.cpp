#include <cstring>

#include "soil/functor.hpp"
#include "soil/error.hpp"
#include "soil/mapper.hpp"
#include "soil/shaper.hpp"
#include "soil/typer.hpp"
#include "soil/grader.hpp"
#include "soil/operator.hpp"
#include "soil/constant.hpp"

#ifdef FUNCTOR_HPP

#define OP_ASSOC(CODE) std::pair<OPCODE,std::string>{CODE, #CODE}

using OpnameMap = EnumMap<OPCODE,std::string>;

const OpnameMap opnames =
{
	OP_ASSOC(CAST),
	OP_ASSOC(ABS),
	OP_ASSOC(NEG),
	OP_ASSOC(NOT),
	OP_ASSOC(SIN),
	OP_ASSOC(COS),
	OP_ASSOC(TAN),
	OP_ASSOC(EXP),
	OP_ASSOC(LOG),
	OP_ASSOC(SQRT),
	OP_ASSOC(ROUND),
	OP_ASSOC(ISMAX),
	OP_ASSOC(POW),
	OP_ASSOC(ADD),
	OP_ASSOC(SUB),
	OP_ASSOC(MUL),
	OP_ASSOC(DIV),
	OP_ASSOC(EQ),
	OP_ASSOC(NE),
	OP_ASSOC(GT),
	OP_ASSOC(LT),
	OP_ASSOC(BINO),
	OP_ASSOC(UNIF),
	OP_ASSOC(NORM),
	OP_ASSOC(TRANSPOSE),
	OP_ASSOC(FLIP),
	OP_ASSOC(ARGMAX),
	OP_ASSOC(RMAX),
	OP_ASSOC(RSUM),
	OP_ASSOC(EXPAND),
	OP_ASSOC(N_ELEMS),
	OP_ASSOC(N_DIMS),
	OP_ASSOC(MATMUL)
};

std::string opname (OPCODE opcode)
{
	auto it = opnames.find(opcode);
	if (opnames.end() == it)
	{
		return "BAD_OP";
	}
	return it->second;
}

Nodeptr Functor::get (std::vector<Nodeptr> args, OPCODE opcode)
{
	return Nodeptr(new Functor(args, opcode));
}

std::shared_ptr<char> Functor::calculate (void)
{
	std::shared_ptr<char> out = make_data(type_size(type_) * shape_.n_elems());
	get_op(opcode_, type_)(out.get(), shape_, args_);
	return out;
}

Nodeptr Functor::gradient (Nodeptr& leaf) const
{
	if (leaf.get() == this)
	{
		return get_one(shape_, type_);
	}
	return get_grader(opcode_)(args_, leaf);
}

Shape Functor::shape (void) const
{
	return shape_;
}

Functor::Functor (std::vector<Nodeptr> args, OPCODE opcode) :
	args_(args), opcode_(opcode)
{
	std::vector<iNode*> raw;
	std::vector<DTYPE> types;
	for (Nodeptr& arg : args_)
	{
		raw.push_back(arg.get());
		types.push_back(arg->type());
	}
	if (std::any_of(raw.begin(), raw.end(),
		[](iNode* arg)
		{
			return nullptr == arg;
		}))
	{
		handle_error("creating functor with null argument");
	}
	shape_ = get_shaper(opcode)(raw);
	type_ = get_typer(opcode_)(types);
}

static inline Shape swap_shape(Nodeptr& arg, CoordOp& op)
{
	std::vector<DimT> dims = arg->shape().as_list();
	op(dims);
	return Shape(dims);
}

Nodeptr Copyover::get (Nodeptr& arg, CoordOp swapdim)
{
	return Nodeptr(new Copyover(arg, swapdim));
}

std::shared_ptr<char> Copyover::calculate (void)
{
	std::shared_ptr<char> src = arg_->calculate();
	DTYPE outtype = type();
	uint8_t bsize = type_size(outtype);
	Shape destshape = shape();
	Shape srcshape = arg_->shape();
	NElemT n = srcshape.n_elems();

	std::shared_ptr<char> out = make_data(
		type_size(outtype) * destshape.n_elems());
	char* destdata = out.get();
	char* srcdata = src.get();

	// apply transformation
	std::vector<DimT> coords;
	for (NElemT srci = 0; srci < n; ++srci)
	{
		coords = coordinate(srcshape, srci);
		swapdim_(coords);
		NElemT desti = index(destshape, coords);
		std::memcpy(destdata + desti * bsize,
			srcdata + srci * bsize, bsize);
	}

	return out;
}

Nodeptr Copyover::gradient (Nodeptr& leaf) const
{
	return arg_->gradient(leaf);
}

Shape Copyover::shape (void) const
{
	return shape_;
}

Copyover::Copyover (Nodeptr& arg, CoordOp swapdim) :
	shape_(swap_shape(arg, swapdim)), arg_(arg), swapdim_(swapdim) {}

CoordOp dim_swap (std::pair<uint8_t,uint8_t> dims)
{
	return [dims](std::vector<DimT>& coords)
	{
		std::swap(coords[dims.first], coords[dims.second]);
	};
}

Nodeptr group (Nodeptr a)
{
	return ShapeTransform::get(a, Shape(std::vector<Shape>{a->shape()}));
}

Nodeptr transpose (Nodeptr a)
{
	return Functor::get({a}, OPCODE::TRANSPOSE);
}

Nodeptr transpose (Nodeptr a, CoordOp swapdim)
{
	return Copyover::get(a, swapdim);
}

Nodeptr operator + (Nodeptr a, Nodeptr b)
{
	return Functor::get({a, b}, OPCODE::ADD);
}

Nodeptr operator * (Nodeptr a, Nodeptr b)
{
	return Functor::get({a, b}, OPCODE::MUL);
}

Nodeptr matmul (Nodeptr a, Nodeptr b)
{
	return Functor::get({a, b}, OPCODE::MATMUL);
}

#endif
