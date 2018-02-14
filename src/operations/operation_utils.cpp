//
//  operation_utils.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-09-07.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/operations/operation_utils.hpp"

#ifdef TENNCOR_OPERATION_UTILS_HPP

namespace nnet
{

static inline AUDSET_T aud_intersects (const std::vector<inode*>& srcs)
{
    AUDSET_T auds = srcs[0]->get_audience();
    // get intersection of all source audiences
    for (size_t i = 1; i < srcs.size(); i++)
    {
        AUDSET_T inner = srcs[i]->get_audience();
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
	AUDSET_T auds = src->get_audience();
    for (iobserver* o : auds)
    {
        iconnector* aud = static_cast<iconnector*>(o);
        std::vector<inode*> args = aud->get_arguments();
        if (args.size() == 1 && opname == args[0]->get_label())
        {
            return aud;
        }
    }
    return nullptr;
}

inode* ordered_parent (std::vector<inode*> srcs, std::string opname)
{
    // assert srcs.size() > 0
    AUDSET_T auds = aud_intersects(srcs);
    for (iobserver* o : auds)
    {
        iconnector* aud = static_cast<iconnector*>(o);
        if (opname == aud->get_label() && 
			std::equal(srcs.begin(), srcs.end(), 
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
    AUDSET_T auds = aud_intersects(srcs);
    for (iobserver* o : auds)
    {
        iconnector* aud = static_cast<iconnector*>(o);
        std::vector<inode*> args = aud->get_arguments();
        std::vector<inode*> diff;
        std::set_difference(
            args.begin(), args.end(),
            srcs.begin(), srcs.end(), 
            std::inserter(diff, diff.begin()));
        if (opname == aud->get_label() && 
			args.size() == srcs.size() && 
            0 == diff.size())
        {
            return aud;
        }
    }
    return nullptr;
}

}

#endif
