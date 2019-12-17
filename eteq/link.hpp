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

#ifndef ETEQ_LINK_HPP
#define ETEQ_LINK_HPP

namespace eteq
{

template <typename T>
struct Functor;

/// Implementation of iEigenEdge using node as tensor wrapper
template <typename T>
struct iLink : public marsh::iAttributed
{
	virtual ~iLink (void) = default;

	iLink<T>* clone (void) const
	{
		return this->clone_impl();
	}

	T* data (void) const
	{
		return (T*) get_tensor()->data();
	}

	teq::Shape link_shape (void) const // todo: abstract this
	{
		return get_tensor()->shape();
	}

	teq::Shape shape (void) const
	{
		return get_tensor()->shape();
	}

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

template <typename T>
LinkptrT<T> data_link (teq::TensptrT data);

}

#endif // ETEQ_LINK_HPP
