//
/// link.hpp
/// eteq
///
/// Purpose:
/// Typed Eigen implementation of eigen iEigenEdge<T>
///

#include "estd/estd.hpp"

#include "marsh/attrs.hpp"

#include "teq/itensor.hpp"

#include "eigen/generated/dtype.hpp"
#include "eigen/eigen.hpp"
#include "eigen/edge.hpp"

#include "eteq/signature.hpp"

#ifndef ETEQ_LINK_HPP
#define ETEQ_LINK_HPP

namespace eteq
{

template <typename T>
struct Functor;

/// Implementation of iEigenEdge using node as tensor wrapper
template <typename T>
struct iLink : public eigen::iEigenEdge<T>, public iSignature, public marsh::iAttributed
{
	virtual ~iLink (void) = default;

	iLink<T>* clone (void) const
	{
		return this->clone_impl();
	}

	/// Implementation of iEigenEdge<T>
	teq::Shape shape (void) const override
	{
		return this->get_tensor()->shape();
	}

	virtual bool has_data (void) const = 0;

	virtual teq::TensptrT get_tensor (void) const = 0;

protected:
	virtual iLink<T>* clone_impl (void) const = 0;

	virtual void subscribe (Functor<T>* parent) = 0;

	virtual void unsubscribe (Functor<T>* parent) = 0;

	friend struct Functor<T>;
};

template <typename T>
using LinkptrT = std::shared_ptr<iLink<T>>;

/// Type of typed functor arguments
template <typename T>
using LinksT = std::vector<LinkptrT<T>>;

template <typename T>
LinkptrT<T> to_link (teq::TensptrT tens);

}

#endif // ETEQ_LINK_HPP
