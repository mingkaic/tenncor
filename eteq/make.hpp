
#include "eigen/packattr.hpp"

#include "eteq/constant.hpp"
#include "eteq/variable.hpp"
#include "eteq/functor.hpp"
#include "eteq/layer.hpp"

#ifndef ETEQ_CONVERT_HPP
#define ETEQ_CONVERT_HPP

namespace eteq
{

/// Return link of tens according to builders in specified converter
template <typename T>
LinkptrT<T> to_link (teq::TensptrT tens)
{
	if (nullptr == tens)
	{
		return nullptr;
	}
	if (auto leaf = std::dynamic_pointer_cast<iLeaf<T>>(tens))
	{
		return std::make_shared<LeafLink<T>>(leaf);
	}
	else if (auto func = std::dynamic_pointer_cast<Functor<T>>(tens))
	{
		return std::make_shared<FuncLink<T>>(func);
	}
	else if (auto layer = std::dynamic_pointer_cast<Layer<T>>(tens))
	{
		return std::make_shared<LayerLink<T>>(layer);
	}
	return nullptr;
}

/// Return variable node given scalar and shape
template <typename T>
VarptrT<T> make_variable_scalar (T scalar,
	teq::Shape shape, std::string label)
{
	if (label.empty())
	{
		label = fmts::to_string(scalar);
	}
	std::vector<T> data(shape.n_elems(), scalar);
	return VarptrT<T>(Variable<T>::get(data.data(), shape, label));
}

/// Return zero-initialized variable node of specified shape
template <typename T>
VarptrT<T> make_variable (teq::Shape shape, std::string label)
{
	return make_variable_scalar<T>(0, shape, label);
}

/// Return variable node filled with scalar matching link shape
template <typename T>
VarptrT<T> make_variable_like (T scalar,
	LinkptrT<T> link, std::string label)
{
	return make_variable_scalar(scalar, link->shape(), label);
}

/// Return variable node given raw array and shape
template <typename T>
VarptrT<T> make_variable (T* data, teq::Shape shape, std::string label)
{
	return VarptrT<T>(Variable<T>::get(data, shape, label));
}

/// Return constant node given scalar and shape
template <typename T>
LinkptrT<T> make_constant_scalar (T scalar, teq::Shape shape)
{
	std::vector<T> data(shape.n_elems(), scalar);
	return to_link<T>(teq::TensptrT(Constant<T>::get(data.data(), shape)));
}

/// Return constant node filled with scalar matching link shape
template <typename T>
LinkptrT<T> make_constant_like (T scalar, LinkptrT<T> link)
{
	return make_constant_scalar(scalar, link->shape());
}

/// Return constant node given raw array and shape
template <typename T>
LinkptrT<T> make_constant (T* data, teq::Shape shape)
{
	return to_link<T>(teq::TensptrT(
		Constant<T>::get(data, shape)));
}

/// Return functor node given opcode and node arguments
template <typename T, typename ...ARGS>
LinkptrT<T> make_functor (egen::_GENERATED_OPCODE opcode, LinksT<T> links, ARGS... vargs)
{
	if (links.empty())
	{
		logs::fatalf("cannot %s without arguments", egen::name_op(opcode).c_str());
	}
	marsh::Maps attrs;
	eigen::pack_attr(attrs, vargs...);
	return to_link<T>(Functor<T>::get(opcode, links, std::move(attrs)));
}

}

#endif // ETEQ_CONVERT_HPP
