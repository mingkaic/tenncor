//
//  functor.cpp
//  mold
//

#include "mold/functor.hpp"

#ifdef MOLD_FUNCTOR_HPP

namespace mold
{

Functor::Functor (std::vector<iNode*> args, OperateIO fwd, GradF bwd) :
    args_(args), fwd_(fwd), bwd_(bwd)
{
    size_t nargs = args.size();
    for (size_t i = 0; i < nargs; ++i)
    {
        args[i]->add(this);
    }
}

bool Functor::has_data (void) const
{
    return nullptr == cache_;
}

clay::State Functor::get_state (void) const
{
    if (nullptr == cache_)
    {
        throw std::exception(); // todo: add context
    }
    return cache_->get_state();
}

NodePtrT Functor::derive (NodeRefT wrt)
{
    return bwd_(wrt, args_);
}

void Functor::initialize (void)
{
    if (false == std::all_of(args_.begin(), args_.end(),
    [](iNode*& arg)
    {
        return arg->has_data();
    }))
    {
        return;
    }
    
    std::vector<clay::State> inputs(args_.size());
    std::transform(args_.begin(), args_.end(), inputs.begin(),
    [](iNode* arg) -> clay::State
    {
        return arg->get_state();
    });
    fwd_.args_ = inputs;
    cache_ = fwd_.get();

    for (Functor* aud : audience_)
    {
        aud->initialize();
    }
}

void Functor::update (void)
{
    if (false == cache_->read_from(fwd_))
    {
        throw std::exception(); // todo: add context
    }
    for (Functor* aud : audience_)
    {
        aud->update();
    }
}

std::vector<iNode*> Functor::get_args (void) const
{
    return args_;
}

}

#endif
