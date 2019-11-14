#include "pbm/marshal.hpp"

#ifdef PBM_MARSHAL_HPP

namespace pbm
{

void marshal_attrs (PbAttrMapT& out, const marsh::Maps& attrs)
{
	for (auto& apair : attrs.contents_)
	{
		if (typeid(marsh::NumArray<double>).
			hash_code() != apair.second->class_code())
		{
			continue;
		}
		cortenn::ArrayAttrs attrs;
		auto& contents = static_cast<
			const marsh::NumArray<double>*>(apair.second.get())->contents_;
		for (double e : contents)
		{
			attrs.add_values(e);
		}
		out.insert({apair.first, attrs});
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
		out.contents_.emplace(pbpair.first, marsh::ObjptrT(out_arr));
	}
}

teq::Shape get_shape (const cortenn::Source& source)
{
	const auto& pb_slist = source.shape();
	std::vector<teq::DimT> slist(pb_slist.begin(), pb_slist.end());
	return teq::Shape(slist);
}

}

#endif
