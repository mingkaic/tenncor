#include "tenncor/eteq/make.hpp"

#ifdef ETEQ_MAKE_HPP

namespace eteq
{

#define _CHOOSE_TYPER(OPCODE)\
typecode = TypeParser<OPCODE>().dtype(attrs, dtypes);

#define _CHOOSE_FUNCTYPE(REALTYPE)\
out = make_tfuncattr<REALTYPE>(opcode, children, attrs);

teq::TensptrT make_funcattr (egen::_GENERATED_OPCODE opcode,
	teq::TensptrsT children, marsh::Maps& attrs)
{
	teq::TensptrT out;
	egen::_GENERATED_DTYPE typecode = egen::BAD_TYPE;
	DTypesT dtypes;
	dtypes.reserve(children.size());
	std::transform(children.begin(), children.end(), std::back_inserter(dtypes),
	[](teq::TensptrT child)
	{
		return (egen::_GENERATED_DTYPE) child->get_meta().type_code();
	});
	OPCODE_LOOKUP(_CHOOSE_TYPER, opcode);
	TYPE_LOOKUP(_CHOOSE_FUNCTYPE, typecode);
	return out;
}

#undef _CHOOSE_TYPER

#undef _CHOOSE_FUNCTYPE

}

#endif
