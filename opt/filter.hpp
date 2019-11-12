///
/// rmdups.hpp
/// opt
///
/// Purpose:
/// Define TEQ functor duplication removal algorithm
///

#include "opt/optimize.hpp"

#ifndef OPT_RMDUPS_HPP
#define OPT_RMDUPS_HPP

namespace opt
{

using EqualF = std::function<bool(teq::TensptrT,teq::TensptrT)>;

using CalcCvsF = std::function<CversionsT(teq::TensptrSetT)>;

/// Delete and update equivalent functor and leaves
void remove_duplicates (teq::TensptrsT& roots, EqualF equals);

void constant_funcs (teq::TensptrsT& roots, CalcCvsF calc_funcs);

}

#endif // OPT_RMDUPS_HPP
