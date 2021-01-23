
#ifndef MARSH_MOCK_MARSHALER_HPP
#define MARSH_MOCK_MARSHALER_HPP

#include "internal/marsh/marsh.hpp"

#include "gmock/gmock.h"

struct MockMarsh final : public marsh::iMarshaler
{
	MOCK_METHOD1(marshal, void(const marsh::String&));
	MOCK_METHOD1(marshal, void(const marsh::iNumber&));
	MOCK_METHOD1(marshal, void(const marsh::iArray&));
	MOCK_METHOD1(marshal, void(const marsh::iTuple&));
	MOCK_METHOD1(marshal, void(const marsh::Maps&));
};

#endif // MARSH_MOCK_MARSHALER_HPP
