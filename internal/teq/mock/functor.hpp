
#ifndef TEQ_MOCK_FUNCTOR_HPP
#define TEQ_MOCK_FUNCTOR_HPP

#include "internal/teq/teq.hpp"

#include "gmock/gmock.h"

using ::testing::_;
using ::testing::Const;
using ::testing::Return;

struct MockFunctor final : public teq::iFunctor
{
	MOCK_CONST_METHOD0(to_string, std::string(void));
	MOCK_METHOD0(device, teq::iDeviceRef&(void));
	MOCK_CONST_METHOD0(device, const teq::iDeviceRef&(void));
	MOCK_CONST_METHOD0(get_meta, const teq::iMetadata&(void));
	MOCK_CONST_METHOD0(shape, teq::Shape(void));
	MOCK_CONST_METHOD0(clone_impl, teq::iTensor*(void));

	MOCK_CONST_METHOD0(get_opcode, teq::Opcode(void));
	MOCK_CONST_METHOD0(get_args, teq::TensptrsT(void));
	MOCK_METHOD2(update_child, void(teq::TensptrT,size_t));

	MOCK_CONST_METHOD0(ls_attrs, types::StringsT(void));

	MOCK_CONST_METHOD1(get_attr, const marsh::iObject*(const std::string&));

	MOCK_METHOD1(get_attr, marsh::iObject*(const std::string&));

	MOCK_METHOD(void, add_attr, (const std::string& attr_key, marsh::ObjptrT&& attr_val), (override));

	MOCK_METHOD1(rm_attr, void(const std::string&));

	MOCK_CONST_METHOD0(size, size_t(void));
};

using MockFuncptrT = std::shared_ptr<MockFunctor>;

MockFuncptrT make_fnc (std::string opname, size_t opcode, teq::TensptrsT args);

#endif // TEQ_MOCK_FUNCTOR_HPP
