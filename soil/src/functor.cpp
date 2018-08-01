#include <cstring>

#include "soil/functor.hpp"
#include "soil/error.hpp"
#include "soil/mapper.hpp"
#include "soil/shaper.hpp"
#include "soil/typer.hpp"
#include "soil/grader.hpp"
#include "soil/operator.hpp"

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

DataSource Functor::calculate (void)
{
	std::vector<OpArg> args;
	std::vector<DTYPE> types;
	for (Nodeptr& arg : args_)
	{
		DataSource ds = arg->calculate();
		args.push_back(OpArg{ds, arg->shape()});
		types.push_back(ds.type());
	}
	DTYPE outtype = get_typer(opcode_)(types);
	DataSource out{outtype, shape_.n_elems()};
	OpArg dest{out, shape_};
	get_op(opcode_, outtype)(dest, args);
	return out;
}

Nodeptr Functor::gradient (Nodeptr& leaf) const
{
	if (leaf.get() == this)
	{
		throw std::bad_function_call(); // unimplemented
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
	std::vector<iNode*> raw(args.size());
	std::transform(args.begin(), args.end(), raw.begin(),
		[](Nodeptr& arg) { return arg.get(); });
	if (std::any_of(raw.begin(), raw.end(),
		[](iNode* arg)
		{
			return nullptr == arg;
		}))
	{
		handle_error("creating functor with null argument");
	}
	shape_ = get_shaper(opcode)(raw);
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

DataSource Copyover::calculate (void)
{
	DataSource src = arg_->calculate();
	DTYPE outtype = src.type();
	uint8_t bsize = type_size(outtype);
	Shape destshape = shape();
	Shape srcshape = arg_->shape();
	DataSource out{outtype, srcshape.n_elems()};
	NElemT n = srcshape.n_elems();

	char* destdata = out.data();
	char* srcdata = src.data();

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
