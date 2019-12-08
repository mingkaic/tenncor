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

const std::string shaper_key = "shape";

const std::string coorder_key = "coorder";

std::vector<teq::CDimT> get_coorder (const marsh::Maps& attrs);

std::vector<teq::CDimT> get_coorder (const teq::iFunctor* func);

}

#endif // EIGEN_EDGE_HPP
