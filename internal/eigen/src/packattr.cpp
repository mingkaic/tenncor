#include "internal/eigen/packattr.hpp"

#ifdef EIGEN_PACKATTR_HPP

namespace eigen
{

std::string Packer<egen::_GENERATED_DTYPE>::key_ = "dtype";

std::string Packer<PairVecT<teq::DimT>>::key_ = "dimension_pairs";

std::string Packer<PairVecT<teq::RankT>>::key_ = "rank_pairs";

std::string Packer<teq::DimsT>::key_ = "dimensions";

std::string Packer<teq::RanksT>::key_ = "ranks";

std::string Packer<std::set<teq::RankT>>::key_ = "rank_set";

std::string Packer<teq::RankT>::key_ = "rank";

std::string Packer<teq::Shape>::key_ = "shape";

std::string Packer<teq::TensptrT>::key_ = "tensor";

void pack_attr (marsh::iAttributed&) {}

OptDimsT unpack_extend (teq::Shape inshape, const marsh::iAttributed& attrib)
{
	teq::DimsT bcast;
	Packer<teq::DimsT> dimpacker;
	Packer<teq::TensptrT> tenspacker;
	if (nullptr != attrib.get_attr(dimpacker.get_key()))
	{
		dimpacker.unpack(bcast, attrib);
	}
	else if (nullptr != attrib.get_attr(tenspacker.get_key()))
	{
		teq::TensptrT tens;
		tenspacker.unpack(tens, attrib);
		auto target = tens->shape();
		for (teq::RankT i = 0; i < teq::rank_cap; ++i)
		{
			teq::DimT tdim = target.at(i);
			if (inshape.at(i) != tdim)
			{
				bcast.push_back(tdim);
			}
			else
			{
				bcast.push_back(1);
			}
		}
	}
	else
	{
		return OptDimsT();
	}
	return bcast;
}

}

#endif
