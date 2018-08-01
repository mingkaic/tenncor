#include "soil/inode.hpp"
#include "soil/error.hpp"

#ifdef NODE_HPP

iNode::~iNode (void) = default;

Nodeptr::Nodeptr (iNode* node) :
	ptr_(node)
{
	if (nullptr == node)
	{
		handle_error("init nodeptr with nullptr");
	}
}

iNode* Nodeptr::operator -> (void)
{
	return ptr_.get();
}

const iNode* Nodeptr::operator -> (void) const
{
	return ptr_.get();
}

iNode* Nodeptr::get (void) const
{
	return ptr_.get();
}

#endif
