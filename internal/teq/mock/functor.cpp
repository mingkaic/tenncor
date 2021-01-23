#include "internal/teq/mock/functor.hpp"

#ifdef TEQ_MOCK_FUNCTOR_HPP

MockFuncptrT make_fnc (std::string opname, size_t opcode, teq::TensptrsT args)
{
	auto out = std::make_shared<MockFunctor>();
	EXPECT_CALL(*out, to_string()).WillRepeatedly(Return(opname));
	EXPECT_CALL(*out, get_opcode()).WillRepeatedly(Return(teq::Opcode{opname, opcode}));
	EXPECT_CALL(*out, get_args()).WillRepeatedly(Return(args));
	EXPECT_CALL(*out, size()).WillRepeatedly(Return(0));
	EXPECT_CALL(*out, ls_attrs()).WillRepeatedly(Return(types::StringsT{}));
	EXPECT_CALL(*out, get_attr(_)).WillRepeatedly(Return(nullptr));
	EXPECT_CALL(Const(*out), get_attr(_)).WillRepeatedly(Return(nullptr));
	return out;
}

#endif
