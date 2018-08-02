#include <unordered_map>

#include "soil/data.hpp"
#include "soil/shape.hpp"
#include "soil/type.hpp"

#ifndef INODE_HPP
#define INODE_HPP

struct Nodeptr;

struct Session;

struct iNode
{
	virtual ~iNode (void);

	virtual std::shared_ptr<char> calculate (Session& sess) = 0;

	virtual Nodeptr gradient (Nodeptr& leaf) const = 0;

	virtual Shape shape (void) const = 0;

	virtual DTYPE type (void) const = 0;
};

struct Session
{
	std::unordered_map<iNode*,std::shared_ptr<char> > pool_;
};

struct Nodeptr
{
	Nodeptr (iNode* node);

	iNode* operator -> (void);

	const iNode* operator -> (void) const;

	iNode* get (void) const;

	std::shared_ptr<iNode> ptr_;
};

#endif /* INODE_HPP */
