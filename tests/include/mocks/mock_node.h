//
// Created by Mingkai Chen on 2017-03-16.
//

#ifndef TENNCOR_MOCK_NODE_H
#define TENNCOR_MOCK_NODE_H

#include <algorithm>

#include "tests/include/utils/util_test.h"

#include "include/graph/inode.hpp"

using namespace nnet;


// not a substitute for leaves...
// WARNING: deletes data
class mock_node : public inode
{
public:
	mock_node (std::string name = "") : inode(name) {}
	mock_node (const mock_node& other) : inode(other) {}
	mock_node (mock_node&& other) : inode(std::move(other)) {}
	mock_node& operator = (const mock_node& other)
	{
		inode::operator = (other);
		return *this;
	}
	mock_node& operator = (mock_node&& other)
	{
		inode::operator = (std::move(other));
		return *this;
	}

	~mock_node (void)
	{
		if (data_) delete data_;
	}

	tensor* data_ = nullptr;

	virtual bool eval (idata_dest& dest) { return data_; }
	virtual tensorshape get_shape (void) const { return data_->get_shape(); }
	virtual std::unordered_set<ileaf*> get_leaves (void) const { return std::unordered_set<ileaf*>{}; }
	virtual varptr derive (inode* wrt) { return wrt == this ? varptr(this) : varptr(); }
	virtual bool read_proto (const tenncor::tensor_proto&) { return false; }

	inode* expose_leaf (inode* source, variable* leaf) const
	{
		return this->take_gradient(source, leaf);
	}

	virtual size_t get_depth (void) const
	{
		return 0;
	}

protected:
	virtual inode* clone_impl (void) const { return new mock_node(*this); }
	virtual inode* move_impl (void) { return new mock_node(std::move(*this)); }
	virtual const tensor* get_eval (void) const { return data_; }
	virtual inode* get_gradient (variable*) { return nullptr; }
};


#endif //TENNCOR_MOCK_NODE_H

