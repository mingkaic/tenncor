//
//  constant.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/graph/constant.hpp"

#ifdef TENNCOR_CONSTANT_HPP

namespace nnet
{

static std::unordered_map<size_t,constant*> cregistry;

static std::unordered_map<constant*,size_t> cbwds;

constant* find_const (size_t key)
{
	auto it = cregistry.find(key);
	if (cregistry.end() != it)
	{
		return it->second;
	}
	return nullptr;
}

bool dangling (constant* key)
{
	return cbwds.end() != cbwds.find(key);
}

void register_const (size_t key, constant* cons)
{
	cregistry[key] = cons;
	cbwds[cons] = key;
}

tensor* constant::get_tensor (void)
{
	return data_.get();
}

varptr constant::derive (inode*)
{
	return nullptr;
}


constant::~constant (void)
{
	auto it = cbwds.find(this);
	if (cbwds.end() != it)
	{
		cregistry.erase(it->second);
		cbwds.erase(it);
	}
}


constant::constant (tenncor::tensor_proto& proto_src,
	std::string label) :
inode(label), data_(std::make_unique<tensor>(proto_src)) {}

nnet::NODE_TYPE constant::node_type (void) const
{
	return tenncor::node_proto::CONSTANT;
}

void constant::serialize_detail (google::protobuf::Any* proto_dest)
{
	tenncor::tensor_proto tens;
	assert(nullptr != data_ && data_->serialize(tens));
	proto_dest->PackFrom(tens);
}


constant::constant (tensor* data, std::string name) :
	inode(name), data_(std::unique_ptr<tensor>(data)) {}

void constant::death_on_noparent (void)
{
	if (this->get_audience().empty())
	{
		delete this;
	}
}

inode* constant::clone_impl (void) const
{
	return nullptr;
}

inode* constant::move_impl (void)
{
	return nullptr;
}

}

#endif
