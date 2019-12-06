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
template <typename T>
struct Edge final : public eigen::iEigenEdge<T>
{
	Edge (NodeptrT<T> node) :
		node_(node)
	{
		if (node_ == nullptr)
		{
			logs::fatal("cannot map a null node");
		}
	}

	/// Implementation of iEdge
	teq::Shape shape (void) const override
	{
		return node_->shape();
	}

	/// Implementation of iEdge
	teq::TensptrT get_tensor (void) const override
	{
		return node_->get_tensor();
	}

	/// Implementation of iEdge
	void get_attrs (marsh::Maps& out) const override {}

	/// Implementation of iEigenEdge<T>
	T* data (void) const override
	{
		return node_->data();
	}

	void set_node (NodeptrT<T> node)
	{
		node_ = node;
	}

	NodeptrT<T> get_node (void) const
	{
		return node_;
	}

private:
	/// Tensor reference
	NodeptrT<T> node_;
};

/// Type of typed functor arguments
template <typename T>
using ArgsT = std::vector<Edge<T>>;

}

#endif // ETEQ_EDGE_HPP
