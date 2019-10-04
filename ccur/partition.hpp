///
/// partition.hpp
/// ccur
///
/// Purpose:
/// Implement the algorithm to separate all nodes under a series of graphs
/// into k groups as to minimize the size of each group while ensuring the
/// parents of every node is found under the same group
///

#include "teq/traveler.hpp"

#ifndef CCUR_PARTITION_HPP
#define CCUR_PARTITION_HPP

namespace ccur
{

/// Groups of functors
using PartGroupsT = std::vector<std::vector<teq::iFunctor*>>;

/// Map functor opcode to the operation's weight value
using OpWeightT = std::unordered_map<size_t,double>;

/// Return k groups of graphs under roots given some weight
PartGroupsT k_partition (teq::TensptrsT roots, size_t k, OpWeightT weights = OpWeightT());

}

#endif // CCUR_PARTITION_HPP
