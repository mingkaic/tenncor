#include "sand/node.hpp"

#ifdef SAND_NODE_HPP

Node::~Node (void) = default;

Shape Node::shape (void) const
{
	return info_.shape_;
}

DTYPE Node::type (void) const
{
	return info_.type_;
}

Node::Node (Meta info) : info_(info) {}

#endif
