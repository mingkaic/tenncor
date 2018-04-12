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


// varptr constant::get_generic (std::string data, TENS_TYPE type)
// {
// 	// make sure data isn't huge before attempting hash
// 	assert(data.size() == type_size(type));
// 	size_t key = ((size_t) type) ^ std::hash<std::string>(data);
// 	constant* cons = find_const(key);
// 	if (nullptr == cons)
// 	{
// 		tensorshape shape = std::vector<size_t>{1};
// 		const_init ci(data, type);
// 		tensor* data = new tensor(shape);
// 		data->read_from(ci);
// 		cons = new constant(data, nnutils::formatter() << scalar);
// 		register_const(key, cons);
// 	}
// 	return cons;
// }

varptr constant::get (tenncor::TensorPb& proto_src, std::string label)
{
	// look in cache if scalar
	constant* cons;
	if (proto_src.alloced_shape_size() == 1 && proto_src.alloced_shape(0) == 1)
	{
		TENS_TYPE type = proto_src.type();
		std::shared_ptr<void> data = deserialize_data(proto_src.data(), type);
		size_t key = ((size_t) type) ^ std::hash<std::string>()(
			std::string((char*) data.get(), type_size(type)));
		cons = find_const(key);
		if (nullptr == cons)
		{
			cons = new constant(new tensor(proto_src), label);
			register_const(key, cons);
		}
	}
	else
	{
		cons = new constant(new tensor(proto_src), label);
	}
	return cons;
}

NODE_TYPE constant::node_type (void) const
{
	return CONSTANT_T;
}

void constant::serialize_detail (google::protobuf::Any* proto_dest) const
{
	tenncor::TensorPb tens;
	assert(nullptr != data_ && data_->serialize(tens));
	proto_dest->PackFrom(tens);
}


tensor* constant::get_tensor (void)
{
	return data_.get();
}

varptr constant::derive (inode*)
{
	return nullptr;
}


constant::constant (tensor* data, std::string name) :
	inode(name), data_(std::unique_ptr<tensor>(data)) {}

constant::~constant (void)
{
	auto it = cbwds.find(this);
	if (cbwds.end() != it)
	{
		cregistry.erase(it->second);
		cbwds.erase(it);
	}
}

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
