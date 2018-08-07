#include <functional>

#include "sand/node.hpp"
#include "sand/opcode.hpp"
#include "sand/preop.hpp"

#ifndef SOIL_FUNCTOR_HPP
#define SOIL_FUNCTOR_HPP

using CoordOp = std::function<void(std::vector<DimT>&)>;

struct Functor final : public Node
{
	static Nodeptr get (std::vector<Nodeptr> args,
		iPreOperator& preop, OPCODE opcode);

	std::shared_ptr<char> calculate (Pool& pool) override;

	Nodeptr gradient (Nodeptr& leaf) const override;

	std::vector<iNode*> get_refs (void) const;

	OPCODE get_opcode (void) const
	{
		return opcode_;
	}

	MetaEncoder get_meta (void) const
	{
		return encoder_;
	}

private:
	Functor (std::vector<Nodeptr> args,
		iPreOperator& preop, OPCODE opcode);

	std::vector<Nodeptr> args_;
	MetaEncoder encoder_;
	OPCODE opcode_;
};

struct ShapeTransform final : public iNode
{
	static Nodeptr get (Nodeptr& arg, Shape shape)
	{
		return new ShapeTransform(arg, shape);
	}

	std::shared_ptr<char> calculate (Pool& pool) override
	{
		return arg_->calculate(pool);
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

#endif /* SOIL_FUNCTOR_HPP */
