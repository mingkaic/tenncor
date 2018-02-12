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

inode* single_parent (inode* src, std::string opname)
{
	AUD_SET auds = src->get_audience();
    for (iobserver* o : auds)
    {
        iconnector* aud = static_cast<iconnector*>(o);
        std::vector<inode*> args = aud->get_arguments();
        if (args.size() == 1 && opname == args->get_label())
        {
            return aud;
        }
    }
}

inode* ordered_parent (std::vector<inode*> srcs, std::string opname)
{
    // assert srcs.size() > 0
    AUD_SET auds = srcs[0]->get_audience();
    // get intersection of all source audiences
    for (size_t i = 1; i < srcs.size(); i++)
    {
        AUD_SET inner = srcs[i]->get_audience();
        std::remove_if(auds.begin(), auds.end(), [&inner](iobserver* o)
        {
            return inner.end() == inner.find(o);
        });
    }
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
    AUD_SET auds = srcs[0]->get_audience();
    std::vector<inode*> diff(srcs.size() * 2);
    auto it = diff.begin();
    // get intersection of all source audiences
    for (size_t i = 1; i < srcs.size(); i++)
    {
        AUD_SET inner = srcs[i]->get_audience();
        std::remove_if(auds.begin(), auds.end(), [&inner](iobserver* o)
        {
            return inner.end() == inner.find(o);
        });
    }
    for (iobserver* o : auds)
    {
        iconnector* aud = static_cast<iconnector*>(o);
        std::vector<inode*> args = aud->get_arguments();
        if (opname == aud->get_label() && 
			args.size() == srcs.size() &&
            it == std::set_difference(
                args.begin(), args.end(), 
                srcs.begin(), srcs.end(), it))
        {
            return aud;
        }
    }
    return nullptr;
}

}

#endif
