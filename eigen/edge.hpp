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

const std::string shaper_key = "shape";

const std::string coorder_key = "coorder";

std::vector<teq::CDimT> get_coorder (const teq::iEdge& edge);

template <typename T>
using EigenEdgesT = std::vector<std::reference_wrapper<const eigen::iEigenEdge<T>>>;

}

#endif // EIGEN_EDGE_HPP
