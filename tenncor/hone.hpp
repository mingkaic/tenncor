
#ifndef TENNCOR_HONE_HPP
#define TENNCOR_HONE_HPP

#include "tenncor/distr.hpp"
#include "tenncor/hone/hone.hpp"
#include "tenncor/hone/hosvc/hosvc.hpp"

namespace tcr
{

void optimize (std::string filename,
    const global::CfgMapptrT& ctx = global::context());

}

#endif // TENNCOR_HONE_HPP
