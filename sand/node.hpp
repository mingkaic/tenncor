#include "sand/inode.hpp"
#include "sand/meta.hpp"

#ifndef SAND_NODE_HPP
#define SAND_NODE_HPP

struct Node : public iNode
{
	virtual ~Node (void);

	Shape shape (void) const override;

	DTYPE type (void) const override;

protected:
	Node (Meta info);

	Meta info_;
};

#endif /* SAND_NODE_HPP */
