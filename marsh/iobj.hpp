#include <functional>

#include "estd/estd.hpp"
#include "fmts/fmts.hpp"

#include "marsh/imarshal.hpp"

#ifndef MARSH_IOBJ_HPP
#define MARSH_IOBJ_HPP

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

	template <typename SUB, typename std::enable_if<
		std::is_base_of<iObject,SUB>::value>::type* = nullptr>
	SUB* cast (void)
	{
		if (typeid(SUB).hash_code() == this->class_code())
		{
			return static_cast<SUB*>(this);
		}
		return nullptr;
	}

	template <typename SUB, typename std::enable_if<
		std::is_base_of<iObject,SUB>::value>::type* = nullptr>
	const SUB* cast (void) const
	{
		if (typeid(SUB).hash_code() == this->class_code())
		{
			return static_cast<const SUB*>(this);
		}
		return nullptr;
	}

protected:
	virtual iObject* clone_impl (void) const = 0;
};

using ObjptrT = std::unique_ptr<iObject>;

}

#endif // MARSH_IOBJ_HPP
