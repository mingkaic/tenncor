//
//  constant.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/graph/leaf/constant.hpp"

#ifdef TENNCOR_CONSTANT_HPP

namespace nnet
{

tensor* constant::get_tensor (void)
{
	return data_.get();
}

varptr constant::derive (inode*)
{
	return nullptr;
}


constant::constant (tenncor::tensor_proto& proto_src,
	std::string label, std::string uid) :
inode(label, uid), data_(std::make_unique<tensor>(proto_src)) {}

NODE_TYPE constant::node_type (void) const
{
	return CONSTANT_T;
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
	if (this->get_audience().size())
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
