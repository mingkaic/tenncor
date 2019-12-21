
#include "eigen/packattr.hpp"

#include "eteq/constant.hpp"
#include "eteq/variable.hpp"
#include "eteq/functor.hpp"

#ifndef ETEQ_CONVERT_HPP
#define ETEQ_CONVERT_HPP

namespace eteq
{

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
	ETensor<T> link, std::string label)
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
ETensor<T> make_constant_scalar (T scalar, teq::Shape shape)
{
	std::vector<T> data(shape.n_elems(), scalar);
	return ETensor<T>(teq::TensptrT(Constant<T>::get(data.data(), shape)));
}

/// Return constant node filled with scalar matching link shape
template <typename T>
ETensor<T> make_constant_like (T scalar, ETensor<T> link)
{
	return make_constant_scalar(scalar, link->shape());
}

/// Return constant node given raw array and shape
template <typename T>
ETensor<T> make_constant (T* data, teq::Shape shape)
{
	return ETensor<T>(teq::TensptrT(
		Constant<T>::get(data, shape)));
}

/// Return functor node given opcode and node arguments
template <typename T, typename ...ARGS>
ETensor<T> make_functor (egen::_GENERATED_OPCODE opcode, ETensorsT<T> links, ARGS... vargs)
{
	if (links.empty())
	{
		logs::fatalf("cannot %s without arguments", egen::name_op(opcode).c_str());
	}
	marsh::Maps attrs;
	eigen::pack_attr(attrs, vargs...);

	if (links.empty())
	{
		logs::fatalf("cannot perform `%s` without arguments",
			egen::name_op(opcode).c_str());
	}
	return ETensor<T>(Functor<T>::get(opcode,
		teq::TensptrsT(links.begin(), links.end()), std::move(attrs)));
}

template <typename T>
ETensor<T> make_layer (teq::Opcode opcode, ETensor<T> input, ETensor<T> output)
{
	egen::_GENERATED_DTYPE tcode = egen::get_type<T>();
	if (tcode != input->type_code())
	{
		logs::fatalf("incompatible tensor types %s and %s: "
			"cross-type functors not supported yet",
			egen::name_type(tcode).c_str(),
			input->type_label().c_str());
	}

	static_cast<teq::iFunctor*>(output.get())->add_attr(teq::layer_key,
		std::make_unique<teq::LayerObj>(opcode.name_, input));
	return output;
}

}

#endif // ETEQ_CONVERT_HPP
