
#ifndef TEQ_MOCK_DERIVE_HPP
#define TEQ_MOCK_DERIVE_HPP

#include "internal/teq/teq.hpp"

#include "gmock/gmock.h"

struct MockDerivativeFunc final : public teq::iDerivativeFuncs
{
	MOCK_CONST_METHOD3(lderive, teq::TensptrT(teq::FuncptrT,teq::TensptrT,size_t));

	MOCK_CONST_METHOD1(get_const_one, teq::TensptrT(teq::iTensor&));

	MOCK_CONST_METHOD1(get_const_zero, teq::TensptrT(teq::iTensor&));

	MOCK_CONST_METHOD1(add, teq::TensptrT(teq::TensptrsT));
};

#endif // TEQ_MOCK_DERIVE_HPP
