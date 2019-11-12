#ifndef MARSH_MARSHAL_HPP
#define MARSH_MARSHAL_HPP

namespace marsh
{

struct iNumber;

struct iArray;

struct Maps;

struct iMarshaler
{
	virtual ~iMarshaler (void) = default;

	virtual void marshal (const iNumber& num) = 0;

	virtual void marshal (const iArray& arr) = 0;

	virtual void marshal (const Maps& mm) = 0;
};

}

#endif // MARSH_MARSHAL_HPP
