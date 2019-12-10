//
/// link.hpp
/// eteq
///
/// Purpose:
/// Typed Eigen implementation of teq iEdge
///

#include "estd/estd.hpp"

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
struct iLink : public eigen::iEigenEdge<T>, public iSignature<T>
{
	virtual ~iLink (void) = default;

	iLink<T>* clone (void) const
	{
		return this->clone_impl();
	}

	/// Implementation of iEdge
	teq::Shape shape (void) const override
	{
		return this->get_tensor()->shape();
	}

	virtual bool has_data (void) const = 0;

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

/// Function for building a link from tensor
template <typename T>
using LinkBuilderF = std::function<LinkptrT<T>(teq::TensptrT)>;

template <typename T>
struct iLeaf;

template <typename T>
struct iFunctor;

template <typename T>
struct LinkConverter final : public teq::iTraveler
{
	/// Implementation of iTraveler
	void visit (teq::iLeaf* leaf) override;

	/// Implementation of iTraveler
	void visit (teq::iFunctor* func) override;

	std::unordered_map<teq::iTensor*,LinkBuilderF<T>> builders_;
};

/// Return link of tens according to builders in specified converter
template <typename T>
LinkptrT<T> to_link (teq::TensptrT tens)
{
	if (nullptr == tens)
	{
		return nullptr;
	}
	LinkConverter<T> converter;
	tens->accept(converter);
	return converter.builders_.at(tens.get())(tens);
}

}

#endif // ETEQ_LINK_HPP
