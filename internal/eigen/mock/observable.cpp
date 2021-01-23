#include "internal/eigen/mock/observable.hpp"

#ifdef EIGEN_MOCK_OBSERVABLE_HPP

MockObsptrT make_obs (std::string opname, size_t opcode, teq::TensptrsT args)
{
	auto out = std::make_shared<MockObservable>(args);
	EXPECT_CALL(*out, to_string()).WillRepeatedly(Return(opname));
	EXPECT_CALL(*out, get_opcode()).WillRepeatedly(Return(teq::Opcode{opname, opcode}));
	EXPECT_CALL(*out, get_args()).WillRepeatedly(Return(args));
	return out;
}

MockObsptrT make_obs (std::string opname, size_t opcode,
	teq::TensptrsT args, marsh::Maps&& attrs)
{
	auto out = std::make_shared<MockObservable>(args, std::move(attrs));
	EXPECT_CALL(*out, to_string()).WillRepeatedly(Return(opname));
	EXPECT_CALL(*out, get_opcode()).WillRepeatedly(Return(teq::Opcode{opname, opcode}));
	EXPECT_CALL(*out, get_args()).WillRepeatedly(Return(args));
	return out;
}

#endif
