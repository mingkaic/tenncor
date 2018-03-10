//
// Created by Mingkai Chen on 2017-03-16.
//

#include "mocker/mocker.hpp"

#include "graph/inode.hpp"

#ifndef TENNCOR_MOCK_NODE_H
#define TENNCOR_MOCK_NODE_H

namespace testutils
{

// not a substitute for leaves...
// WARNING: deletes data
class mock_node final : public nnet::inode
{
public:
	mock_node (std::string label) : nnet::inode(label) {}

	virtual std::unordered_set<nnet::inode*> get_leaves (void) const { return {}; }

	virtual nnet::tensor* get_tensor (void) { return nullptr; }

	virtual nnet::varptr derive (nnet::inode* wrt) { return nullptr; }

protected:
	virtual nnet::inode* clone_impl (void) const
	{ return new mock_node(*this); }

	virtual nnet::inode* move_impl (void)
	{ return new mock_node(std::move(*this)); }
};

}

#endif //TENNCOR_MOCK_NODE_H
