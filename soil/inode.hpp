#include <memory>

#include "soil/data.hpp"

#ifndef NODE_HPP
#define NODE_HPP

struct Nodeptr;

struct iNode
{
	virtual ~iNode (void);

	virtual DataSource calculate (void) = 0;

	virtual Nodeptr gradient (Nodeptr& leaf) const = 0;

	virtual Shape shape (void) const = 0;
};

struct Nodeptr
{
	Nodeptr (iNode* node);

	iNode* operator -> (void);

	const iNode* operator -> (void) const;

	iNode* get (void) const;

	std::shared_ptr<iNode> ptr_;
};

#endif /* NODE_HPP */
