#include "marsh/objs.hpp"

#ifdef MARSH_OBJS_HPP

namespace marsh
{

void get_attrs (Maps& mvalues, const iAttributed& attributed)
{
	auto keys = attributed.ls_attrs();
	for (const std::string& key : keys)
	{
		if (auto obj = attributed.get_attr(key))
		{
			mvalues.add_attr(key, ObjptrT(obj->clone()));
		}
	}
}

}

#endif
