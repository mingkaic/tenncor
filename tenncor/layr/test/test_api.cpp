
#ifndef DISABLE_LAYER_API_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "testutil/tutil.hpp"

#include "internal/teq/mock/mock.hpp"

#include "tenncor/layr/layer.hpp"


using ::testing::Invoke;


TEST(LAYER, MakeGetInput)
{
	teq::Shape shape({3, 2});
	auto x = make_var(shape, "x");
	auto x2 = make_var(shape, "x2");
	auto x3 = make_var(shape, "x3");

	auto f = make_fnc("ADD", 5, teq::TensptrsT{x, x2});
	auto layer_root = make_fnc("MUL", 6, teq::TensptrsT{f,x3});
	marsh::ObjptrT cap_val;
	EXPECT_CALL(*layer_root, add_attr("layer", _)).WillOnce(Invoke(
	[&](const std::string& key, marsh::ObjptrT&& attr_val)
	{
		cap_val = std::move(attr_val);
	}));
	auto layer = layr::make_layer(layer_root, "example", x);
	auto cap = dynamic_cast<teq::LayerObj*>(cap_val.get());
	ASSERT_NE(nullptr, cap);
	EXPECT_STREQ("example", cap->to_string().c_str());
	EXPECT_EQ(x.get(), cap->get_tensor().get());
	EXPECT_EQ(layer, layer_root);

	EXPECT_CALL(*layer_root, ls_attrs()).WillRepeatedly(Return(types::StringsT{teq::layer_attr}));
	EXPECT_CALL(*layer_root, size()).WillRepeatedly(Return(1));
	EXPECT_CALL(*layer_root, get_attr(teq::layer_attr)).WillRepeatedly(Return(cap));
	EXPECT_EQ(x.get(), layr::get_input(layer).get());
}


#endif // DISABLE_LAYER_API_TEST
