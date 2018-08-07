#include <memory>
#include <unordered_map>

#include "sand/shape.hpp"
#include "sand/type.hpp"

#ifndef SAND_INODE_HPP
#define SAND_INODE_HPP

struct Nodeptr;

struct Pool;

struct iNode
{
	virtual ~iNode (void);

	// expensive: unnecessary passing shared_ptr, todo: use pool
	virtual std::shared_ptr<char> calculate (Pool& pool) = 0;

	virtual Nodeptr gradient (Nodeptr& leaf) const = 0;

	virtual Shape shape (void) const = 0;

	virtual DTYPE type (void) const = 0;
};

struct Pool
{
	std::unordered_map<iNode*,std::shared_ptr<char> > data_;
};

struct Nodeptr
{
	Nodeptr (iNode* node);

	virtual ~Nodeptr (void) = default;

	iNode* operator -> (void);

	const iNode* operator -> (void) const;

	iNode* get (void) const;

	std::weak_ptr<iNode> ref (void) const
	{
		return ptr_;
	}

protected:
	std::shared_ptr<iNode> ptr_;
};

#endif /* SAND_INODE_HPP */
