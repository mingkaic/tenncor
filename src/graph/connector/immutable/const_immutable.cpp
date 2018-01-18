//
//  const_immutable.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-09-17.
//  Copyright © 2017 Mingkai Chen. All rights reserved.
//

#include "include/graph/connector/immutable/const_immutable.hpp"
#include "include/graph/leaf/constant.hpp"
#include "include/tensor/actors/tens_elem_uni.hpp"

#ifdef TENNCOR_IMMUTABLE_HPP

namespace nnet
{

const_immutable* const_immutable::get (inode* x)
{
	return new const_immutable(x);
}

const_immutable::const_immutable (inode* x) :
	immutable(std::vector<inode*>{x},
	[](std::vector<tensorshape> shapes) { return shapes[0]; },
	new actor_func(
	CONN_ACTOR([](out_wrapper<void>& dest, 
		std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_pipein<double>(dest, srcs);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_pipein<signed>(dest, srcs);
			default:
			break;
		}
		return nullptr;
	})),
	[](std::vector<std::pair<inode*,inode*>>)
	{
		return constant::get_shared_zero();
	}, "const_immutable")
{
	this->gcache_.clear();
	this->jacobians_.clear();
}

}

#endif
