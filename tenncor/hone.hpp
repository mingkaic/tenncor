
#ifndef TENNCOR_HONE_HPP
#define TENNCOR_HONE_HPP

#include "tenncor/hone/hone.hpp"

namespace tcr
{

void optimize (std::string filename,
    const global::CfgMapptrT& ctx = global::context());

}

#endif // TENNCOR_HONE_HPP
