//
//  operation_utils.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-09-07.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/operate/operation_utils.hpp"

#ifdef TENNCOR_OPERATION_UTILS_HPP

namespace nnet
{

static inline inode* find_audience (inode* node, OPCODE opcode, 
	std::function<bool(functor*)> is_target)
{
	AUDMAP_T audmap = node->get_audience();
	for (auto audpair : audmap)
	{
		functor* aud = dynamic_cast<functor*>(audpair.first);
		if (aud && opcode == aud->get_opcode() && is_target(aud))
		{
			return aud;
		}
	}
	return nullptr;
}

inode* single_parent (inode* src, OPCODE opcode)
{
	return find_audience(src, opcode, 
	[](functor* arg)
	{
		return 1 == arg->get_arguments().size();
	});
}

inode* ordered_parent (std::vector<inode*> srcs, OPCODE opcode)
{
	// assert(srcs.size() > 0);
	return find_audience(srcs[0], opcode, 
	[&](functor* arg)
	{
		std::vector<inode*> args = arg->get_arguments();
		return std::equal(srcs.begin(), srcs.end(), 
			args.begin(), args.end());
	});
}

inode* unordered_parent (std::vector<inode*> srcs, OPCODE opcode)
{
	// assert(srcs.size() > 1);
	std::unordered_set<inode*> srcset(srcs.begin(), srcs.end());
	return find_audience(srcs[0], opcode, 
	[&](functor* arg)
	{
		std::vector<inode*> args = arg->get_arguments();
		std::unordered_set<inode*> argset(args.begin(), args.end());
		return std::equal(srcset.begin(), srcset.end(), 
			argset.begin(), argset.end());
	});
}

}

#endif
