#include "eigen/packattr.hpp"

#include "eteq/constant.hpp"
#include "eteq/variable.hpp"
#include "eteq/functor.hpp"
#include "eteq/funcopt.hpp"

#ifndef ETEQ_MAKE_HPP
#define ETEQ_MAKE_HPP

namespace eteq
{

/// Return variable node given scalar and shape
template <typename T>
EVariable<T> make_variable_scalar (T scalar,
	teq::Shape shape, std::string label)
{
	if (label.empty())
	{
		label = fmts::to_string(scalar);
	}
	std::vector<T> data(shape.n_elems(), scalar);
	return EVariable<T>(VarptrT<T>(Variable<T>::get(data.data(), shape, label)));
}

/// Return zero-initialized variable node of specified shape
template <typename T>
EVariable<T> make_variable (teq::Shape shape, std::string label)
{
	return make_variable_scalar<T>(0, shape, label);
}

/// Return variable node filled with scalar matching link shape
template <typename T>
EVariable<T> make_variable_like (
	T scalar, teq::TensptrT like, std::string label)
{
	return make_variable_scalar(scalar, like->shape(), label);
}

/// Return variable node given raw array and shape
template <typename T>
EVariable<T> make_variable (T* data, teq::Shape shape, std::string label)
{
	return EVariable<T>(VarptrT<T>(Variable<T>::get(data, shape, label)));
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
ETensor<T> make_constant_like (T scalar, teq::TensptrT like)
{
	return make_functor<T>(::egen::EXTEND,teq::TensptrsT{
		make_constant_scalar<T>(scalar, teq::Shape())
	}, (teq::TensptrT) like);
}

/// Return constant node given raw array and shape
template <typename T>
ETensor<T> make_constant (T* data, teq::Shape shape)
{
	return ETensor<T>(teq::TensptrT(Constant<T>::get(data, shape)));
}

#define CHOOSE_FUNCOPT(OPCODE)\
redundant = FuncOpt<OPCODE>().is_redundant(attrs, shapes);

template <typename T>
ETensor<T> make_funcattr (egen::_GENERATED_OPCODE opcode,
	const teq::TensptrsT& children, marsh::Maps& attrs)
{
	if (children.empty())
	{
		logs::fatalf("cannot %s without arguments", egen::name_op(opcode).c_str());
	}

	teq::ShapesT shapes;
	shapes.reserve(children.size());
	std::transform(children.begin(), children.end(),
		std::back_inserter(shapes),
		[](teq::TensptrT child)
		{
			return child->shape();
		});
	bool redundant = false;
	OPCODE_LOOKUP(CHOOSE_FUNCOPT, opcode)
	if (redundant)
	{
		return children.front();
	}
	return ETensor<T>(teq::TensptrT(Functor<T>::get(
		opcode, children, std::move(attrs))));
}

#undef CHOOSE_FUNCOPT

/// Return functor node given opcode and node arguments
template <typename T, typename ...ARGS>
ETensor<T> make_functor (egen::_GENERATED_OPCODE opcode,
	const teq::TensptrsT& children, ARGS... vargs)
{
	marsh::Maps attrs;
	eigen::pack_attr(attrs, vargs...);

	return make_funcattr<T>(opcode, children, attrs);
}

template <typename T>
ETensor<T> make_layer (std::string layername,
	teq::TensptrT input, teq::FuncptrT output)
{
	output->add_attr(teq::layer_key,
		std::make_unique<teq::LayerObj>(layername, input));
	return ETensor<T>(output);
}

}

#endif // ETEQ_MAKE_HPP
