//
//  operation_utils.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-09-07.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/operate/operation_utils.hpp"
#include "include/graph/func/functor.hpp"

#ifdef TENNCOR_OPERATION_UTILS_HPP

namespace nnet
{

static inline std::list<iobserver*> aud_intersects (
	const std::vector<inode*>& srcs, std::string opname)
{
	std::list<iobserver*> auds;
	for (auto audpair : srcs[0]->get_audience())
	{
		functor* icon = dynamic_cast<functor*>(audpair.first);
		if (icon && opname == icon->get_label())
		{
			auds.push_back(audpair.first);
		}
	}
	// get intersection of all source audiences
	for (size_t i = 1; i < srcs.size(); i++)
	{
		AUDMAP_T inner = srcs[i]->get_audience();
		auto it = auds.begin();
		while (it != auds.end())
		{
			if (inner.end() == inner.find(*it))
			{
				auto er = it;
				++it;
				auds.erase(er);
			}
			else
			{
				++it;
			}
		}
	}
	return auds;
}

inode* single_parent (inode* src, std::string opname)
{
	AUDMAP_T auds = src->get_audience();
	for (auto audpair : auds)
	{
		if (functor* aud = dynamic_cast<functor*>(audpair.first))
		{
			std::vector<inode*> args = aud->get_arguments();
			if (args.size() == 1 && opname == args[0]->get_label())
			{
				return aud;
			}
		}
	}
	return nullptr;
}

inode* ordered_parent (std::vector<inode*> srcs, std::string opname)
{
	// assert srcs.size() > 0
	auto auds = aud_intersects(srcs, opname);
	for (iobserver* o : auds)
	{
		functor* aud = dynamic_cast<functor*>(o);
		if (aud && std::equal(srcs.begin(), srcs.end(), 
			aud->get_arguments().begin()))
		{
			return aud;
		}
	}
	return nullptr;
}

inode* unordered_parent (std::vector<inode*> srcs, std::string opname)
{
	// assert srcs.size() > 0
	auto auds = aud_intersects(srcs, opname);
	for (iobserver* o : auds)
	{
		if (functor* aud = dynamic_cast<functor*>(o))
		{
			return aud;
		}
	}
	return nullptr;
}

}

#endif
