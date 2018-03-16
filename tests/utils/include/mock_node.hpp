//
// Created by Mingkai Chen on 2017-03-16.
//

#include "mocker/mocker.hpp"

#include "graph/inode.hpp"

#ifndef TENNCOR_MOCK_NODE_HPP
#define TENNCOR_MOCK_NODE_HPP

namespace testutils
{

// not a substitute for leaves...
// WARNING: deletes data
class mock_node final : public nnet::inode
{
public:
	mock_node (std::string label);

	mock_node (std::string label, std::string uid);

	virtual std::unordered_set<const nnet::inode*> get_leaves (void) const;

	virtual nnet::tensor* get_tensor (void);

	virtual nnet::varptr derive (nnet::inode* wrt);

protected:
	virtual NODE_TYPE node_type (void) const
	{
		return VARIABLE_T;
	}

	virtual void serialize_detail (google::protobuf::Any* proto_dest) {}

private:
	virtual nnet::inode* clone_impl (void) const;

	virtual nnet::inode* move_impl (void);
};

}

#endif /* TENNCOR_MOCK_NODE_HPP */
