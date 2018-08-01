#include "soil/inode.hpp"

#ifndef FUNCTOR_HPP
#define FUNCTOR_HPP

enum OPCODE
{
	CAST = 0,
	ABS,
	NEG,
	NOT,
	SIN,
	COS,
	TAN,
	EXP,
	LOG,
	SQRT,
	ROUND,
	ISMAX,
	FLIP,
	EXPAND,
	TRANSPOSE,
	N_ELEMS,
	N_DIMS,
	POW,
	ADD,
	SUB,
	MUL,
	DIV,
	EQ,
	NE,
	GT,
	LT,
	BINO,
	UNIF,
	NORM,
	ARGMAX,
	RMAX,
	RSUM,
	MATMUL
};

using CoordOp = std::function<void(std::vector<DimT>&)>;

std::string opname (OPCODE opcode);

struct Functor final : public iNode
{
	static Nodeptr get (std::vector<Nodeptr> args, OPCODE opcode);

	DataSource calculate (void) override;

	Nodeptr gradient (Nodeptr& leaf) const override;

	Shape shape (void) const override;

private:
	Functor (std::vector<Nodeptr> args, OPCODE opcode);

	Shape shape_;

	std::vector<Nodeptr> args_;
	OPCODE opcode_;
};

struct Copyover final : public iNode
{
	static Nodeptr get (Nodeptr& arg, CoordOp swapdim);

	DataSource calculate (void) override;

	Nodeptr gradient (Nodeptr& leaf) const override;

	Shape shape (void) const override;

private:
	Copyover (Nodeptr& arg, CoordOp swapdim);

	Shape shape_;

	Nodeptr arg_;
	CoordOp swapdim_;
};

struct ShapeTransform final : public iNode
{
	static Nodeptr get (Nodeptr& arg, Shape shape)
	{
		return new ShapeTransform(arg, shape);
	}

	DataSource calculate (void) override
	{
		return arg_->calculate();
	}

	Nodeptr gradient (Nodeptr& leaf) const override
	{
		return arg_->gradient(leaf);
	}

	Shape shape (void) const override
	{
		return shape_;
	}

private:
	ShapeTransform (Nodeptr& arg, Shape shape) :
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

	Shape shape_;
	Nodeptr arg_;
};

CoordOp dim_swap (std::pair<uint8_t,uint8_t> dims);

Nodeptr group (Nodeptr a);

Nodeptr transpose (Nodeptr a);

Nodeptr transpose (Nodeptr a, CoordOp dim_op);

Nodeptr operator + (Nodeptr a, Nodeptr b);

Nodeptr operator * (Nodeptr a, Nodeptr b);

Nodeptr matmul (Nodeptr a, Nodeptr b);

#endif /* FUNCTOR_HPP */
