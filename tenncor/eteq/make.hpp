
#ifndef ETEQ_MAKE_HPP
#define ETEQ_MAKE_HPP

#include "tenncor/eteq/constant.hpp"
#include "tenncor/eteq/functor.hpp"
#include "tenncor/eteq/evars.hpp"
#include "tenncor/eteq/caster.hpp"

namespace eteq
{

using DimPairsT = std::pair<teq::DimT,teq::DimT>;

/// Return variable node given raw array and shape
template <typename T>
EVariable make_variable (T* data, teq::Shape shape,
	const std::string& label = "",
	const global::CfgMapptrT& ctx = global::context())
{
	return EVariable(VarptrT(
		Variable::get(data, egen::get_type<T>(), shape, label)), ctx);
}

/// Return variable node given scalar and shape
template <typename T>
EVariable make_variable_scalar (T scalar,
	teq::Shape shape, std::string label = "",
	const global::CfgMapptrT& ctx = global::context())
{
	if (label.empty())
	{
		label = fmts::to_string(scalar);
	}
	std::vector<T> data(shape.n_elems(), scalar);
	return make_variable(data.data(), shape, label, ctx);
}

/// Return zero-initialized variable node of specified shape
template <typename T>
EVariable make_variable (teq::Shape shape,
	std::string label = "",
	const global::CfgMapptrT& ctx = global::context())
{
	return make_variable_scalar<T>(0, shape, label, ctx);
}

/// Return variable node filled with scalar matching link shape
template <typename T>
EVariable make_variable_like (T scalar, teq::TensptrT like,
	std::string label = "",
	const global::CfgMapptrT& ctx = global::context())
{
	return make_variable_scalar(scalar, like->shape(), label, ctx);
}

teq::TensptrT make_funcattr (egen::_GENERATED_OPCODE opcode,
	teq::TensptrsT children, marsh::Maps& attrs);

#define _CHOOSE_FUNCOPT(OPCODE)\
redundant = egen::FuncOpt<OPCODE>().operator()<T>(attrs, children);

#define _CHOOSE_TYPECAST(OPCODE)\
children = TypeCaster<OPCODE>().operator()<T>(children);

template <typename T>
teq::TensptrT make_tfuncattr (egen::_GENERATED_OPCODE opcode,
	teq::TensptrsT children, marsh::Maps& attrs)
{
	if (children.empty())
	{
		global::fatalf("cannot %s without arguments",
			egen::name_op(opcode).c_str());
	}

	bool redundant = false;
	OPCODE_LOOKUP(_CHOOSE_FUNCOPT, opcode)
	if (redundant)
	{
		return children.front();
	}
	OPCODE_LOOKUP(_CHOOSE_TYPECAST, opcode)
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

template <typename T>
teq::TensptrT make_constant_tensor (T* data, teq::Shape shape)
{
	return teq::TensptrT(Constant<T>::get(data, shape));
}

template <typename T>
teq::TensptrT make_constant_tensor (T* data,
	const eigen::OptSparseT& sparse_info, teq::Shape shape)
{
	return teq::TensptrT(Constant<T>::get(data, sparse_info, shape));
}

/// Return constant node given raw array and shape
template <typename T>
ETensor make_constant (T* data, teq::Shape shape,
	const global::CfgMapptrT& ctx = global::context())
{
	return ETensor(make_constant_tensor(data, shape), ctx);
}

#define _CHOOSE_CSTTYPE(REALTYPE){\
std::vector<REALTYPE> tmp(data, data + shape.n_elems());\
cst = make_constant_tensor<REALTYPE>(tmp.data(), shape);\
}

template <typename T>
teq::TensptrT make_constant_tensor (T* data, teq::Shape shape, egen::_GENERATED_DTYPE dtype)
{
	if (egen::get_type<T>() == dtype)
	{
		return make_constant_tensor<T>(data, shape);
	}
	teq::TensptrT cst;
	TYPE_LOOKUP(_CHOOSE_CSTTYPE, dtype);
	return cst;
}

#undef _CHOOSE_CSTTYPE

#define _CHOOSE_SPARSE_CSTTYPE(REALTYPE){\
std::vector<REALTYPE> tmp(data, data + nelems);\
cst = make_constant_tensor<REALTYPE>(tmp.data(), sparse_info, shape);\
}

template <typename T>
teq::TensptrT make_constant_tensor (T* data, teq::Shape shape,
	egen::_GENERATED_DTYPE dtype, const eigen::OptSparseT& sparse_info)
{
	if (egen::get_type<T>() == dtype)
	{
		return make_constant_tensor<T>(data, sparse_info, shape);
	}
	teq::NElemT nelems = shape.n_elems();
	if (sparse_info)
	{
		nelems = sparse_info->non_zeros_;
	}
	teq::TensptrT cst;
	TYPE_LOOKUP(_CHOOSE_SPARSE_CSTTYPE, dtype);
	return cst;
}

#undef _CHOOSE_SPARSE_CSTTYPE

template <typename T>
ETensor make_constant (T* data, teq::Shape shape, egen::_GENERATED_DTYPE dtype,
	const global::CfgMapptrT& ctx = global::context())
{
	return ETensor(make_constant_tensor(data, shape, dtype), ctx);
}

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
		cst = make_constant_scalar<T>(scalar, teq::Shape(), ctx);
	}
	else
	{
		cst = make_constant_scalar<T>(scalar, teq::Shape(), like_type, ctx);
	}
	return ETensor(make_functor(::egen::EXTEND,
		teq::TensptrsT{cst}, (teq::TensptrT) like), ctx);
}

template <typename T>
ETensor make_constant_like_uncast (T scalar, teq::TensptrT like,
	const global::CfgMapptrT& ctx = global::context())
{
	return ETensor(make_functor(::egen::EXTEND,
		teq::TensptrsT{make_constant_scalar<T>(scalar, teq::Shape(), ctx)},
		(teq::TensptrT) like), ctx);
}

}

#endif // ETEQ_MAKE_HPP
