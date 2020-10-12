
#ifndef MARSH_IOBJ_HPP
#define MARSH_IOBJ_HPP

#include <functional>

#include "estd/estd.hpp"
#include "fmts/fmts.hpp"

#include "internal/marsh/imarshal.hpp"

namespace marsh
{

struct iObject
{
	virtual ~iObject (void) = default;

	iObject* clone (void) const
	{
		return clone_impl();
	}

	virtual size_t class_code (void) const = 0;

	virtual std::string to_string (void) const = 0;

	virtual bool equals (const iObject& other) const = 0;

	virtual void accept (iMarshaler& marshaler) const = 0;

protected:
	virtual iObject* clone_impl (void) const = 0;
};

using ObjptrT = std::unique_ptr<iObject>;

}

#endif // MARSH_IOBJ_HPP
