//
//  inode.cpp
//  mold
//

#include "mold/inode.hpp"
#include "mold/functor.hpp"

#ifdef MOLD_INODE_HPP

namespace mold
{

iNode::~iNode (void)
{
    for (Functor* aud : audience_)
    {
        std::vector<iNode*> args = aud->get_args();
        for (iNode*& arg : args)
        {
            arg->del(aud);
        }
        delete aud;
    }
}

AudienceT iNode::get_audience (void) const
{
    return audience_;
}

void iNode::add (Functor* aud)
{
    audience_.emplace(aud);
}

void iNode::del (Functor* aud)
{
    audience_.erase(aud);
}

}

#endif
