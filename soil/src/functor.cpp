#include <cstring>
#include <algorithm>

#include "sand/shaper.hpp"
#include "sand/typer.hpp"
#include "sand/operator.hpp"

#include "soil/data.hpp"
#include "soil/functor.hpp"
#include "soil/grader.hpp"
#include "soil/constant.hpp"

#include "util/error.hpp"

#ifdef FUNCTOR_HPP

static inline std::pair<Shape,DTYPE> shape_typify (
	std::vector<Nodeptr>& args, OPCODE opcode)
{
	std::vector<Shape> shapes;
	std::vector<DTYPE> types;
	for (Nodeptr& arg : args)
	{
		if (nullptr == arg.get())
		{
			handle_error("creating functor with null argument");
		}
		shapes.push_back(arg->shape());
		types.push_back(arg->type());
	}
	Shape shape = get_shaper(opcode)(shapes);
	DTYPE type = get_typer(opcode)(types);
	return {shape, type};
}

Nodeptr Functor::get (std::vector<Nodeptr> args, OPCODE opcode)
{
	return Nodeptr(new Functor(args, opcode));
}

std::shared_ptr<char> Functor::calculate (Session& sess)
{
	std::vector<std::shared_ptr<char> > temp; // todo: remove
	std::vector<NodeInfo> args;
	std::transform(args_.begin(), args_.end(), std::back_inserter(args),
	[&sess, &temp](Nodeptr& arg)
	{
		temp.push_back(arg->calculate(sess));
		return NodeInfo{
			temp.back().get(),
			arg->shape()
		};
	});
	std::shared_ptr<char> out = make_data(nbytes());
	NodeInfo dest{
		out.get(),
		shape_
	};
	get_op(opcode_, type_)(dest, args);
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

Functor::Functor (std::vector<Nodeptr> args, OPCODE opcode) :
	Node(shape_typify(args, opcode)), args_(args), opcode_(opcode) {}

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

std::shared_ptr<char> Copyover::calculate (Session& sess)
{
	std::shared_ptr<char> src = arg_->calculate(sess);
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

ShapeTransform::ShapeTransform (Nodeptr& arg, Shape shape) :
	shape_(shape), arg_(arg)
{
	NElemT nin = shape.n_elems();
	NElemT nout = arg->shape().n_elems();
	if (nin != nout)
	{
		handle_error("shape transform data of incompatible size",
			ErrArg<NElemT>("indata_size", nin),
			ErrArg<NElemT>("outdata_size", nout));
	}
}

#endif
