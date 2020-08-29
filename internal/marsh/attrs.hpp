
#ifndef MARSH_ATTRS_HPP
#define MARSH_ATTRS_HPP

#include "internal/marsh/iobj.hpp"

namespace marsh
{

struct iAttributed
{
	virtual ~iAttributed (void) = default;

	virtual types::StringsT ls_attrs (void) const = 0;

	virtual const iObject* get_attr (const std::string& attr_key) const = 0;

	virtual iObject* get_attr (const std::string& attr_key) = 0;

	virtual void add_attr (const std::string& attr_key, ObjptrT&& attr_val) = 0;

	virtual void rm_attr (const std::string& attr_key) = 0;

	virtual size_t size (void) const = 0;
};

}

#endif // MARSH_ATTRS_HPP
