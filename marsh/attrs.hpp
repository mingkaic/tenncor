#include "marsh/iobj.hpp"

#ifndef MARSH_ATTRS_HPP
#define MARSH_ATTRS_HPP

namespace marsh
{

struct iAttributed
{
	virtual ~iAttributed (void) = default;

	virtual const marsh::iObject* get_attr (std::string attr_name) const = 0;

	virtual std::vector<std::string> ls_attrs (void) const = 0;
};

}

#endif // MARSH_ATTRS_HPP
