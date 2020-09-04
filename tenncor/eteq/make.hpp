
#ifndef ETEQ_MAKE_HPP
#define ETEQ_MAKE_HPP

#include "tenncor/eteq/constant.hpp"
#include "tenncor/eteq/functor.hpp"
#include "tenncor/eteq/evars.hpp"
#include "tenncor/eteq/typer.hpp"

namespace eteq
{

using DimPairsT = std::pair<teq::DimT,teq::DimT>;

// auto cast children to desired output type
template <typename T>
void autocast_children (teq::TensptrsT& children)
{
	marsh::Maps attrs;
	auto type = egen::get_type<T>();
	for (teq::TensptrT& child : children)
	{
		if (child->get_meta().type_code() != type)
		{
			child = teq::TensptrT(Functor<T>::get(
				egen::CAST, {child}, std::move(attrs)));
		}
	}
}

/// Return variable node given scalar and shape
template <typename T>
EVariable<T> make_variable_scalar (T scalar,
	teq::Shape shape, std::string label = "",
	const global::CfgMapptrT& ctx = global::context())
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
EVariable<T> make_variable (teq::Shape shape,
	std::string label = "",
	const global::CfgMapptrT& ctx = global::context())
{
	return make_variable_scalar<T>(0, shape, label, ctx);
}

/// Return variable node filled with scalar matching link shape
template <typename T>
EVariable<T> make_variable_like (T scalar, teq::TensptrT like,
	std::string label = "",
	const global::CfgMapptrT& ctx = global::context())
{
	return make_variable_scalar(scalar, like->shape(), label, ctx);
}

/// Return variable node given raw array and shape
template <typename T>
EVariable<T> make_variable (T* data, teq::Shape shape,
	std::string label = "",
	const global::CfgMapptrT& ctx = global::context())
{
	return EVariable<T>(VarptrT<T>(
		Variable<T>::get(data, shape, label)), ctx);
}

teq::TensptrT make_funcattr (egen::_GENERATED_OPCODE opcode,
	teq::TensptrsT children, marsh::Maps& attrs);

#define _CHOOSE_FUNCOPT(OPCODE)\
redundant = FuncOpt<OPCODE>().is_redundant(attrs, shapes);

template <typename T>
teq::TensptrT make_tfuncattr (egen::_GENERATED_OPCODE opcode,
	teq::TensptrsT children, marsh::Maps& attrs)
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

	autocast_children<T>(children);
	return teq::TensptrT(Functor<T>::get(opcode, children, std::move(attrs)));
}

#undef _CHOOSE_FUNCOPT

/// Return functor node given opcode and node arguments
template <typename ...ARGS>
teq::TensptrT make_functor (egen::_GENERATED_OPCODE opcode,
	const teq::TensptrsT& children, ARGS... vargs)
{
	marsh::Maps attrs;
	eigen::pack_attr(attrs, vargs...);
	return make_funcattr(opcode, children, attrs);
}

template <typename T, typename ...ARGS>
teq::TensptrT make_tfunctor (egen::_GENERATED_OPCODE opcode,
	const teq::TensptrsT& children, ARGS... vargs)
{
	marsh::Maps attrs;
	eigen::pack_attr(attrs, vargs...);
	return make_tfuncattr<T>(opcode, children, attrs);
}

/// Return constant node given raw array and shape
template <typename T>
ETensor make_constant (T* data, teq::Shape shape,
	const global::CfgMapptrT& ctx = global::context())
{
	return ETensor(teq::TensptrT(
		Constant<T>::get(data, shape)), ctx);
}

#define _CHOOSE_CSTTYPE(REALTYPE){\
std::vector<REALTYPE> tmp(data, data + shape.n_elems());\
cst = make_constant<REALTYPE>(tmp.data(), shape, ctx);\
}

template <typename T>
ETensor make_constant (T* data, teq::Shape shape, egen::_GENERATED_DTYPE dtype,
	const global::CfgMapptrT& ctx = global::context())
{
	if (egen::get_type<T>() == dtype)
	{
		return make_constant<T>(data, shape, ctx);
	}
	ETensor cst;
	TYPE_LOOKUP(_CHOOSE_CSTTYPE, dtype);
	return cst;
}

#undef _CHOOSE_CSTTYPE

/// Return constant node given scalar and shape
template <typename T>
ETensor make_constant_scalar (T scalar, teq::Shape shape,
	const global::CfgMapptrT& ctx = global::context())
{
	std::vector<T> data(shape.n_elems(), scalar);
	return make_constant(data.data(), shape, ctx);
}

template <typename T>
ETensor make_constant_scalar (T scalar, teq::Shape shape,
	egen::_GENERATED_DTYPE dtype,
	const global::CfgMapptrT& ctx = global::context())
{
	std::vector<T> data(shape.n_elems(), scalar);
	return make_constant(data.data(), shape, dtype, ctx);
}

/// Return constant node filled with scalar matching link shape
template <typename T>
ETensor make_constant_like (T scalar, teq::TensptrT like,
	const global::CfgMapptrT& ctx = global::context())
{
	auto like_type = (egen::_GENERATED_DTYPE) like->get_meta().type_code();
	ETensor cst;
	if (like_type == egen::get_type<T>())
	{
		cst = make_constant_scalar<T>(scalar, teq::Shape());
	}
	else
	{
		cst = make_constant_scalar<T>(scalar, teq::Shape(), like_type);
	}
	return ETensor(make_functor(::egen::EXTEND,
		teq::TensptrsT{cst}, (teq::TensptrT) like), ctx);
}

template <typename T>
ETensor make_constant_like_uncast (T scalar, teq::TensptrT like,
	const global::CfgMapptrT& ctx = global::context())
{
	return ETensor(make_functor(::egen::EXTEND,
		teq::TensptrsT{make_constant_scalar<T>(scalar, teq::Shape())},
		(teq::TensptrT) like), ctx);
}

}

#endif // ETEQ_MAKE_HPP
