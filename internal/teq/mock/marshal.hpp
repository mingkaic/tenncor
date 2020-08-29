
#include "internal/teq/objs.hpp"

#ifndef TEQ_MOCK_MARSHALER_HPP
#define TEQ_MOCK_MARSHALER_HPP

struct MockTeqMarsh : public teq::iTeqMarshaler
{
	virtual ~MockTeqMarsh (void) = default;

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

	void marshal (const teq::TensorObj& tens) override
	{
		visited_.emplace(&tens);
	}

	void marshal (const teq::LayerObj& layer) override
	{
		visited_.emplace(&layer);
	}

	std::unordered_set<const marsh::iObject*> visited_;
};

#endif // TEQ_MOCK_MARSHALER_HPP
