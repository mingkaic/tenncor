//
//  ioperate_io.cpp
//  mold
//

#include "mold/ioperate_io.hpp"

#ifdef MOLD_OPERATE_IO_HPP

namespace mold
{

iOperateIO::~iOperateIO (void) = default;

iOperateIO* iOperateIO::clone (void) const
{
    return clone_impl();
}

}

#endif
