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
};

template <typename T>
using EEdgeptrT = std::shared_ptr<iEigenEdge<T>>;

template <typename T>
using EEdgesT = std::vector<EEdgeptrT<T>>;

template <typename T>
using EEdgeRefsT = std::vector<std::reference_wrapper<const iEigenEdge<T>>>;

}

#endif // EIGEN_EDGE_HPP
