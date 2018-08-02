#include <functional>

#include "soil/node.hpp"
#include "soil/opcode.hpp"

#ifndef FUNCTOR_HPP
#define FUNCTOR_HPP

using CoordOp = std::function<void(std::vector<DimT>&)>;

struct Functor final : public Node
{
	static Nodeptr get (std::vector<Nodeptr> args, OPCODE opcode);

	std::shared_ptr<char> calculate (Session& sess) override;

	Nodeptr gradient (Nodeptr& leaf) const override;

private:
	Functor (std::vector<Nodeptr> args, OPCODE opcode);

	std::vector<Nodeptr> args_;
	OPCODE opcode_;
};

struct Copyover final : public iNode
{
	static Nodeptr get (Nodeptr& arg, CoordOp swapdim);

	std::shared_ptr<char> calculate (Session& sess) override;

	Nodeptr gradient (Nodeptr& leaf) const override;

	Shape shape (void) const override;

	DTYPE type (void) const override
	{
		return arg_->type();
	}

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

	std::shared_ptr<char> calculate (Session& sess) override
	{
		return arg_->calculate(sess);
	}

	Nodeptr gradient (Nodeptr& leaf) const override
	{
		return arg_->gradient(leaf);
	}

	Shape shape (void) const override
	{
		return shape_;
	}

	DTYPE type (void) const override
	{
		return arg_->type();
	}

private:
	ShapeTransform (Nodeptr& arg, Shape shape);

	Shape shape_;
	Nodeptr arg_;
};

#endif /* FUNCTOR_HPP */
