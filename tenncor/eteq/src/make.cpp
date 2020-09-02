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

}

#endif
