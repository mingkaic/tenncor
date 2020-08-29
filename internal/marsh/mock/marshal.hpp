
#include "internal/marsh/imarshal.hpp"

#ifndef MARSH_MOCK_MARSHALER_HPP
#define MARSH_MOCK_MARSHALER_HPP

struct MockMarsh : public marsh::iMarshaler
{
	virtual ~MockMarsh (void) = default;

	void marshal (const marsh::String& str) override
	{
		visited_.emplace(&str);
	}

	void marshal (const marsh::iNumber& num) override
	{
		visited_.emplace(&num);
	}

	void marshal (const marsh::iArray& arr) override
	{
		visited_.emplace(&arr);
	}

	void marshal (const marsh::iTuple& tup) override
	{
		visited_.emplace(&tup);
	}

	void marshal (const marsh::Maps& mm) override
	{
		visited_.emplace(&mm);
	}

	std::unordered_set<const marsh::iObject*> visited_;
};

#endif // MARSH_MOCK_MARSHALER_HPP
