#include "internal/teq/ifunctor.hpp"

#include "experimental/print/detail.hpp"

#ifdef DBG_DETAIL_HPP

namespace dbg
{

std::string detail_str (marsh::iObject* obj, int64_t attrdepth)
{
	if (attrdepth > 0)
	{
		if (auto tens = dynamic_cast<teq::TensorObj*>(obj))
		{
			return detail_str(tens->get_tensor(), attrdepth - 1);
		}
		else if (auto lay = dynamic_cast<teq::LayerObj*>(obj))
		{
			return lay->to_string() + "->" +
				detail_str(lay->get_tensor(), attrdepth - 1);
		}
	}
	return obj->to_string();
}

std::string detail_str (teq::iTensor* tens, int64_t attrdepth)
{
	if (nullptr == tens)
	{
		return "";
	}
	// format:
	// <string>:<shape-label>[:{<attr-key>:<attr-val>,...}]
	std::string out = tens->to_string() + delim + tens->shape().to_string();
	if (auto f = dynamic_cast<teq::iFunctor*>(tens))
	{
		std::vector<std::string> attrstrs = f->ls_attrs();
		for (const std::string& attrstr : attrstrs)
		{
			attrstr += ":" + detail_str(f->get_attr(attrstr), attrdepth);
		}
		out += delim + "{";
		out += fmts::join(",", attrstrs.begin(), attrstrs.end());
		out += '}';
	}
	return out;
}

}

#endif
