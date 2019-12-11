#include "marsh/objs.hpp"

#ifdef MARSH_OBJS_HPP

namespace marsh
{

void get_attrs (marsh::Maps& mvalues, const iAttributed& attributed)
{
	auto keys = attributed.ls_attrs();
	for (std::string key : keys)
	{
		if (auto obj = attributed.get_attr(key))
		{
			mvalues.add_attr(key, ObjptrT(obj->clone()));
		}
	}
}

}

#endif
