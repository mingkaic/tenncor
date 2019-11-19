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

using CalcCvsF = std::function<teq::TensptrT(teq::FuncptrT)>;

/// Delete and update equivalent functor and leaves
void remove_duplicates (teq::TensptrsT& roots, EqualF equals);

teq::TensptrT constant_func (teq::FuncptrT& func,
	ParentReplF replace, CalcCvsF calc_func);

void constant_funcs (teq::TensptrsT& roots, CalcCvsF calc_func);

}

#endif // OPT_RMDUPS_HPP
