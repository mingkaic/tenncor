//
//  range.cpp
//  mold
//

#include "mold/state_range.hpp"

#ifdef MOLD_STATE_RANGE_HPP

namespace mold
{

StateRange::StateRange (clay::State arg, Range drange) :
    arg_(arg), drange_(drange) {}

char* StateRange::get (void) const
{
    return arg_.get();
}

clay::Shape StateRange::shape (void) const
{
    return arg_.shape_;
}

clay::DTYPE StateRange::type (void) const
{
    return arg_.dtype_;
}

clay::Shape StateRange::inner (void) const
{
    return drange_.apply(arg_.shape_);
}

clay::Shape StateRange::outer (void) const
{
    return drange_.remove(arg_.shape_);
}

}

#endif /* MOLD_STATE_RANGE_HPP */
