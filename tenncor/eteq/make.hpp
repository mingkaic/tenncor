#include "estd/cast.hpp"

#include "tenncor/eteq/constant.hpp"
#include "tenncor/eteq/functor.hpp"
#include "tenncor/eteq/evars.hpp"

#ifndef ETEQ_MAKE_HPP
#define ETEQ_MAKE_HPP

namespace eteq
{

using DimPairsT = std::pair<teq::DimT,teq::DimT>;

/// Return variable node given scalar and shape
template <typename T>
EVariable<T> make_variable_scalar (T scalar,
	teq::Shape shape, std::string label, const global::CfgMapptrT& ctx)
{
	if (label.empty())
	{
		label = fmts::to_string(scalar);
	}
	std::vector<T> data(shape.n_elems(), scalar);
	return EVariable<T>(VarptrT<T>(
		Variable<T>::get(data.data(), shape, label)), ctx);
}

/// Return zero-initialized variable node of specified shape
template <typename T>
EVariable<T> make_variable (teq::Shape shape, std::string label,
	const global::CfgMapptrT& ctx)
{
	return make_variable_scalar<T>(0, shape, label, ctx);
}

/// Return variable node filled with scalar matching link shape
template <typename T>
EVariable<T> make_variable_like (T scalar, teq::TensptrT like,
	std::string label, const global::CfgMapptrT& ctx)
{
	return make_variable_scalar(scalar, like->shape(), label, ctx);
}

/// Return variable node given raw array and shape
template <typename T>
EVariable<T> make_variable (T* data, teq::Shape shape,
	std::string label, const global::CfgMapptrT& ctx)
{
	return EVariable<T>(VarptrT<T>(
		Variable<T>::get(data, shape, label)), ctx);
}

/// Return constant node given scalar and shape
template <typename T>
ETensor<T> make_constant_scalar (T scalar, teq::Shape shape,
	const global::CfgMapptrT& ctx)
{
	std::vector<T> data(shape.n_elems(), scalar);
	return ETensor<T>(teq::TensptrT(
		Constant<T>::get(data.data(), shape)), ctx);
}

/// Return constant node filled with scalar matching link shape
template <typename T>
ETensor<T> make_constant_like (T scalar, teq::TensptrT like,
	const global::CfgMapptrT& ctx)
{
	return make_functor<T>(ctx, ::egen::EXTEND,teq::TensptrsT{
		make_constant_scalar<T>(scalar, teq::Shape())
	}, (teq::TensptrT) like);
}

/// Return constant node given raw array and shape
template <typename T>
ETensor<T> make_constant (T* data, teq::Shape shape,
	const global::CfgMapptrT& ctx)
{
	return ETensor<T>(teq::TensptrT(
		Constant<T>::get(data, shape)), ctx);
}

#define _CHOOSE_FUNCOPT(OPCODE)\
redundant = FuncOpt<OPCODE>().is_redundant(attrs, shapes);

template <typename T>
teq::TensptrT make_funcattr (egen::_GENERATED_OPCODE opcode,
	const teq::TensptrsT& children, marsh::Maps& attrs)
{
	if (children.empty())
	{
		global::fatalf("cannot %s without arguments",
			egen::name_op(opcode).c_str());
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
	OPCODE_LOOKUP(_CHOOSE_FUNCOPT, opcode)
	if (redundant)
	{
		return children.front();
	}
	return teq::TensptrT(Functor<T>::get(
		opcode, children, std::move(attrs)));
}

#undef _CHOOSE_FUNCOPT

/// Return functor node given opcode and node arguments
template <typename T, typename ...ARGS>
teq::TensptrT make_functor (egen::_GENERATED_OPCODE opcode,
	const teq::TensptrsT& children, ARGS... vargs)
{
	marsh::Maps attrs;
	eigen::pack_attr(attrs, vargs...);

	return make_funcattr<T>(opcode, children, attrs);
}

template <typename T, typename ...ARGS>
ETensor<T> make_functor (const global::CfgMapptrT& ctx,
	egen::_GENERATED_OPCODE opcode,
	const teq::TensptrsT& children, ARGS... vargs)
{
	return ETensor<T>(make_functor<T,ARGS...>(opcode, children,
		std::forward<ARGS>(vargs)...), ctx);
}

teq::TensptrT add_dependencies (teq::TensptrT root,
	teq::TensptrsT dependencies);

}

#endif // ETEQ_MAKE_HPP
