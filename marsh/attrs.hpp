#include "marsh/iobj.hpp"

#ifndef MARSH_ATTRS_HPP
#define MARSH_ATTRS_HPP

namespace marsh
{

struct iAttributed
{
	virtual ~iAttributed (void) = default;

	virtual std::vector<std::string> ls_attrs (void) const = 0;

	virtual const iObject* get_attr (const std::string& attr_key) const = 0;

	virtual iObject* get_attr (const std::string& attr_key) = 0;

	virtual void add_attr (const std::string& attr_key, ObjptrT&& attr_val) = 0;

	virtual void rm_attr (const std::string& attr_key) = 0;
};

}

#endif // MARSH_ATTRS_HPP
