#include "tenncor/eteq/make.hpp"

#ifdef ETEQ_MAKE_HPP

namespace eteq
{

static egen::_GENERATED_DTYPE max_precision (const teq::TensptrsT& children)
{
	egen::_GENERATED_DTYPE best_type = egen::BAD_TYPE;
	for (auto child : children)
	{
		auto dtype = (egen::_GENERATED_DTYPE) child->get_meta().type_code();
		if (egen::type_precision(dtype) > egen::type_precision(best_type))
		{
			best_type = dtype;
		}
	}
	return best_type;
}

#define _CHOOSE_FUNCTYPE(REALTYPE)\
out = make_tfuncattr<REALTYPE>(opcode, children, attrs);

teq::TensptrT make_funcattr (egen::_GENERATED_OPCODE opcode,
	teq::TensptrsT children, marsh::Maps& attrs)
{
	teq::TensptrT out;
	auto typecode = max_precision(children);
	TYPE_LOOKUP(_CHOOSE_FUNCTYPE, typecode);
	return out;
}

#undef _CHOOSE_FUNCTYPE

teq::TensptrT add_dependencies (teq::TensptrT root,
	teq::TensptrsT dependencies)
{
	auto f = estd::must_cast<teq::iFunctor>(root.get());
	auto deps_attr = dynamic_cast<teq::TensArrayT*>(
		f->get_attr(dependency_key));
	if (nullptr == deps_attr)
	{
		f->add_attr(dependency_key,
			std::make_unique<teq::TensArrayT>());
		deps_attr = static_cast<teq::TensArrayT*>(
			f->get_attr(dependency_key));
	}
	auto& contents = deps_attr->contents_;
	for (auto& tens : dependencies)
	{
		contents.emplace(contents.end(),
			std::make_unique<teq::TensorObj>(tens));
	}
	return root;
}

}

#endif
