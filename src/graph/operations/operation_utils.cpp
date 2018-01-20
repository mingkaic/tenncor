//
//  operation_utils.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-09-07.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "include/graph/operations/operation_utils.hpp"

#ifdef TENNCOR_OPERATION_UTILS_HPP

namespace nnet
{

inode* unary_parent_search (inode* operand, std::string opname)
{
	std::unordered_set<inode*> audience;
	if (operand->find_audience(opname, audience))
	{
		return *audience.begin();
	}
	return nullptr;
}

inode* ordered_binary_parent_search (inode* a, inode* b, std::string opname)
{
	std::unordered_set<inode*> audience;
	if (a->find_audience(opname, audience))
	{
		// linear search on audience
		for (inode* aud : audience)
		{
			std::vector<inode*> args = aud->get_arguments();
			if (args.size() == 2 && args[0] == a && args[1] == b)
			{
				return aud;
			}
		}
	}
	return nullptr;
}

inode* unordered_binary_parent_search (inode* a, inode* b, std::string opname)
{
	std::unordered_set<inode*> audience;
	if (a->find_audience(opname, audience))
	{
		// linear search on audience
		for (inode* aud : audience)
		{
			std::vector<inode*> args = aud->get_arguments();
			if (args.size() == 2 && (
					(args[0] == a && args[1] == b) ||
					(args[1] == a && args[0] == b)))
			{
				return aud;
			}
		}
	}
	return nullptr;
}

}

#endif
