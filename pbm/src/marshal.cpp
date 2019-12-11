#include "pbm/marshal.hpp"

#ifdef PBM_MARSHAL_HPP

namespace pbm
{

void marshal_attrs (PbAttrMapT& out, const marsh::Maps& attrs)
{
	auto keys = attrs.ls_attrs();
	for (std::string key : keys)
	{
		auto val = attrs.get_attr(key);
		if (typeid(marsh::NumArray<double>).
			hash_code() != val->class_code())
		{
			continue;
		}
		auto& contents = static_cast<
			const marsh::NumArray<double>*>(val)->contents_;
		tenncor::ArrayAttrs pb_attrs;
		for (double e : contents)
		{
			pb_attrs.add_values(e);
		}
		out.insert({key, pb_attrs});
	}
}

void unmarshal_attrs (marsh::Maps& out, const PbAttrMapT& pb_map)
{
	for (const auto& pbpair : pb_map)
	{
		auto& pb_values = pbpair.second.values();
		auto out_arr = new marsh::NumArray<double>();
		for (double e : pb_values)
		{
			out_arr->contents_.push_back(e);
		}
		out.add_attr(pbpair.first, marsh::ObjptrT(out_arr));
	}
}

teq::Shape get_shape (const tenncor::Source& source)
{
	const auto& pb_slist = source.shape();
	std::vector<teq::DimT> slist(pb_slist.begin(), pb_slist.end());
	return teq::Shape(slist);
}

}

#endif
