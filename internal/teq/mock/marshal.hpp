
#ifndef TEQ_MOCK_MARSHALER_HPP
#define TEQ_MOCK_MARSHALER_HPP

#include "internal/teq/teq.hpp"

#include "gmock/gmock.h"

struct MockTeqMarsh final : public teq::iTeqMarshaler
{
	MOCK_METHOD1(marshal, void(const marsh::String&));
	MOCK_METHOD1(marshal, void(const marsh::iNumber&));
	MOCK_METHOD1(marshal, void(const marsh::iArray&));
	MOCK_METHOD1(marshal, void(const marsh::iTuple&));
	MOCK_METHOD1(marshal, void(const marsh::Maps&));

	MOCK_METHOD1(marshal, void(const teq::TensorObj&));

	MOCK_METHOD1(marshal, void(const teq::LayerObj&));
};

#endif // TEQ_MOCK_MARSHALER_HPP
