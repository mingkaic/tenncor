#include "sand/inode.hpp"

#ifndef NODE_HPP
#define NODE_HPP

struct Node : public iNode
{
	virtual ~Node (void) = default;

	Shape shape (void) const override
	{
		return shape_;
	}

	DTYPE type (void) const override
	{
		return type_;
	}

protected:
	Node (std::pair<Shape,DTYPE> stpair) :
		shape_(stpair.first), type_(stpair.second) {}

	size_t nbytes (void) const
	{
		return type_size(type_) * shape_.n_elems();
	}

	Shape shape_;
	DTYPE type_;
};

#endif /* NODE_HPP */
