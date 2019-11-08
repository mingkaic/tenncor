#include "teq/iedge.hpp"

#ifndef EIGEN_EDGE_HPP
#define EIGEN_EDGE_HPP

namespace eigen
{

template <typename T>
struct iEigenEdge : public teq::iEdge
{
	virtual ~iEigenEdge (void) = default;

	virtual T* data (void) const = 0;

	virtual void set_tensor (teq::TensptrT tens) = 0;
};

const std::string coorder_key = "coord";

std::vector<teq::CDimT> get_coorder (const teq::iEdge& edge)
{
	marsh::Maps mvalues;
	edge.get_attrs(mvalues);
	if (false == estd::has(mvalues.contents_, coorder_key))
	{
		logs::fatal("coorder not found");
	}
	auto& coorder = mvalues.contents_.at(coorder_key);
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

template <typename T>
using EigenEdgesT = std::vector<std::reference_wrapper<const eigen::iEigenEdge<T>>>;

}

#endif // EIGEN_EDGE_HPP
