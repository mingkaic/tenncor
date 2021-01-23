#include "internal/teq/mock/leaf.hpp"

#ifdef TEQ_MOCK_LEAF_HPP

void make_var (MockLeaf& out, const teq::Shape& shape, const std::string& label)
{
	EXPECT_CALL(out, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(out, to_string()).WillRepeatedly(Return(label));
	EXPECT_CALL(out, get_usage()).WillRepeatedly(Return(teq::IMMUTABLE));
}

MockLeafptrT make_var (const teq::Shape& shape, const std::string& label)
{
	auto out = std::make_shared<MockLeaf>();
	make_var(*out, shape, label);
	return out;
}

#endif
