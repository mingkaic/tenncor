#include "eigen/packattr.hpp"

#ifdef EIGEN_PACKATTR_HPP

namespace eigen
{

std::string Packer<eigen::PairVecT<teq::DimT>>::key_ = "dimension_pairs";

std::string Packer<eigen::PairVecT<teq::RankT>>::key_ = "rank_pairs";

std::string Packer<std::vector<teq::DimT>>::key_ = "dimensions";

std::string Packer<std::vector<teq::RankT>>::key_ = "ranks";

std::string Packer<std::set<teq::RankT>>::key_ = "rank_set";

std::string Packer<teq::RankT>::key_ = "rank";

std::string Packer<teq::Shape>::key_ = "shape";

void pack_attr (marsh::Maps& attrs) {}

}

#endif
