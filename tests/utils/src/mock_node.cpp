#include "mock_node.hpp"

#ifdef TENNCOR_MOCK_NODE_HPP

namespace testutils
{

mock_node::mock_node (std::string label) : nnet::inode(label) {}

std::unordered_set<const nnet::inode*> mock_node::get_leaves (void) const { return {this}; }

nnet::tensor* mock_node::get_tensor (void) { return nullptr; }

nnet::varptr mock_node::derive (nnet::inode* wrt) { return nullptr; }

nnet::inode* mock_node::clone_impl (void) const
{ return new mock_node(*this); }

nnet::inode* mock_node::move_impl (void)
{ return new mock_node(std::move(*this)); }

}

#endif
