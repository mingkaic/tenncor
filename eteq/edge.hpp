//
/// edge.hpp
/// eteq
///
/// Purpose:
/// Typed Eigen implementation of teq iEdge
///

#include "eigen/operator.hpp"

#include "eteq/inode.hpp"

#ifndef ETEQ_EDGE_HPP
#define ETEQ_EDGE_HPP

namespace eteq
{

/// Implementation of iEigenEdge using node as tensor wrapper
template <typename T, typename C=double>
struct Edge final : public eigen::iEigenEdge<T>
{
	Edge (NodeptrT<T> node) :
		node_(node)
	{
		if (node_ == nullptr)
		{
			logs::fatal("cannot map a null node");
		}
		shape_ = node->shape();
	}

	Edge (NodeptrT<T> node, teq::Shape shape,
		std::vector<C> coords) :
		node_(node), shape_(shape), coords_(coords)
	{
		if (node_ == nullptr)
		{
			logs::fatal("cannot map a null node");
		}
	}

	/// Implementation of iEdge
	teq::Shape shape (void) const override
	{
		return shape_;
	}

	/// Implementation of iEdge
	teq::Shape argshape (void) const override
	{
		return node_->shape();
	}

	/// Implementation of iEdge
	teq::TensptrT get_tensor (void) const override
	{
		return node_->get_tensor();
	}

	/// Implementation of iEdge
	void get_attrs (marsh::Maps& out) const override
	{
		if (false == shape_.compatible_after(node_->shape(), 0))
		{
			auto arr = std::make_unique<marsh::NumArray<double>>();
			arr->contents_ = std::vector<double>(shape_.begin(), shape_.end());
			out.contents_.emplace(eigen::shaper_key, std::move(arr));
		}
		if (coords_.size() > 0)
		{
			auto arr = std::make_unique<marsh::NumArray<double>>();
			arr->contents_ = std::vector<double>(coords_.begin(), coords_.end());
			out.contents_.emplace(eigen::coorder_key, std::move(arr));
		}
	}

	/// Implementation of iEigenEdge<T>
	T* data (void) const override
	{
		return node_->data();
	}

	void set_tensor (teq::TensptrT tens)
	{
		node_ = to_node<T>(tens);
	}

	NodeptrT<T> get_node (void) const
	{
		return node_;
	}

private:
	/// Tensor reference
	NodeptrT<T> node_;

	/// Output shape
	teq::Shape shape_;

	/// Coordinate transformation attributes
	std::vector<C> coords_;
};

/// Type of typed functor arguments
template <typename T>
using ArgsT = std::vector<Edge<T>>;

}

#endif // ETEQ_EDGE_HPP
