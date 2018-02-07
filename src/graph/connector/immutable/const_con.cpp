//
//  const_con.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-09-17.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/graph/connector/immutable/const_con.hpp"
#include "include/graph/leaf/constant.hpp"

#ifdef TENNCOR_CONST_CON_HPP

namespace nnet
{

const_con* const_con::get (inode* x)
{
	return new const_con(x);
}

const_con::const_con (inode* x) :
	linear(std::vector<inode*>{x},
	[](std::vector<tensorshape> shapes) { return shapes[0]; },
	new actor_func(
	CONN_ACTOR([](out_wrapper<void>& dest,
		std::vector<in_wrapper<void> >& srcs,
		TENS_TYPE type) -> itens_actor*
	{
		switch (type)
		{
			case DOUBLE:
				return new tens_pipein<double>(dest, srcs);
			case INT:
				return new tens_pipein<signed>(dest, srcs);
			default:
			break;
		}
		return nullptr;
	})),
	[](std::vector<std::pair<inode*,inode*>>)
	{
		return constant::get_shared_zero();
	}, "const_con")
{
	this->gcache_.clear();
	this->jacobians_.clear();
}

}

#endif
