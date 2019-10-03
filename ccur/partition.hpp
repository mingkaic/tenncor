#include "teq/traveler.hpp"

#ifndef CCE_PARTITION_HPP
#define CCE_PARTITION_HPP

namespace ccur
{

using PartGroupsT = std::vector<std::vector<teq::iFunctor*>>;

using OpWeightT = std::unordered_map<size_t,double>;

PartGroupsT k_partition (teq::TensptrsT roots, size_t k, OpWeightT weights = OpWeightT());

}

#endif // CCE_PARTITION_HPP
