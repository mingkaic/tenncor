#include "internal/teq/itensor.hpp"

#ifndef DBG_DETAIL_HPP
#define DBG_DETAIL_HPP

namespace dbg
{

std::string detail_str (marsh::iObject* obj, int64_t attrdepth = 0);

std::string detail_str (teq::iTensor* tens, int64_t attrdepth = 0);

}

#endif // DBG_DETAIL_HPP
