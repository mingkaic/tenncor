#include "eigen/edge.hpp"

#ifdef EIGEN_EDGE_HPP

namespace eigen
{

std::vector<teq::CDimT> get_coorder (const marsh::Maps& attrs)
{
	if (false == estd::has(attrs.contents_, coorder_key))
	{
		logs::fatal("coorder not found");
	}
	auto& coorder = attrs.contents_.at(coorder_key);
	if (coorder->class_code() != typeid(marsh::NumArray<teq::CDimT>).hash_code())
	{
		logs::fatal("cannot find array coorder");
	}
	auto& ccontent = static_cast<marsh::NumArray<teq::CDimT>*>(
		coorder.get())->contents_;
	std::vector<teq::CDimT> out;
	out.reserve(ccontent.size());
	for (auto& val : ccontent)
	{
		out.push_back(val);
	}
	return out;
}

std::vector<teq::CDimT> get_coorder (const teq::iEdge& edge)
{
	marsh::Maps mvalues;
	edge.get_attrs(mvalues);
	return get_coorder(mvalues);
}

}

#endif
