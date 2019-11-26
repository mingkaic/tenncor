
#ifndef DISABLE_LOCATOR_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "teq/mock/leaf.hpp"
#include "teq/mock/functor.hpp"

#include "tag/locator.hpp"

#include "tag/mock/tag.hpp"


TEST(LOCATOR, DisplayLocation)
{
	tag::TagRegistry treg;

	EXPECT_STREQ("<nil>", tag::display_location(nullptr).c_str());

	teq::TensptrT tens = std::make_shared<MockTensor>(teq::Shape({3, 4, 5}), "leaf");
	EXPECT_STREQ(">leaf<:[3\\4\\5\\1\\1\\1\\1\\1]{}", tag::display_location(tens).c_str());

	treg.add_tag(tens, std::make_unique<MockTag>(tid, "special_tag"));
	EXPECT_STREQ(">leaf<:[3\\4\\5\\1\\1\\1\\1\\1]{special_tag:[A\\B\\C\\1]}",
		tag::display_location(tens, {}, treg).c_str());

	teq::TensptrT tens2 = std::make_shared<MockTensor>(teq::Shape({1, 22, 7}), "leaf2");
	auto vfunc = std::make_shared<MockFunctor>(teq::TensptrsT{tens2, tens}, teq::Opcode{"Quack", 0});
	auto cfunc = std::make_shared<MockFunctor>(teq::TensptrsT{vfunc, vfunc, tens}, teq::Opcode{"Parent", 0});

	EXPECT_STREQ(
		"Quack:[1\\22\\7\\1\\1\\1\\1\\1]\n"
		" `-->leaf<:[3\\4\\5\\1\\1\\1\\1\\1]{special_tag:[A\\B\\C\\1]}",
		tag::display_location(tens, {vfunc}, treg).c_str());

	EXPECT_STREQ(
		"Parent:[1\\22\\7\\1\\1\\1\\1\\1]\n"
		"Quack:[1\\22\\7\\1\\1\\1\\1\\1]\n"
		" `-->leaf<:[3\\4\\5\\1\\1\\1\\1\\1]{special_tag:[A\\B\\C\\1]}",
		tag::display_location(tens, {cfunc}, treg).c_str());

	EXPECT_STREQ(
		"Parent:[1\\22\\7\\1\\1\\1\\1\\1]\n"
		" `-->Quack<:[1\\22\\7\\1\\1\\1\\1\\1]{}\n"
		"     `--(leaf2):[1\\22\\7\\1\\1\\1\\1\\1]\n"
		"     `--(leaf):[3\\4\\5\\1\\1\\1\\1\\1]",
		tag::display_location(vfunc, {cfunc}, treg).c_str());

	treg.add_tag(vfunc, std::make_unique<MockTag>(tid, "normal_funk"));
	EXPECT_STREQ(
		"Parent:[1\\22\\7\\1\\1\\1\\1\\1]\n"
		" `-->Quack<:[1\\22\\7\\1\\1\\1\\1\\1]{normal_funk:[A\\B\\C\\1]}\n"
		"     `--(leaf2):[1\\22\\7\\1\\1\\1\\1\\1]\n"
		"     `--(leaf):[3\\4\\5\\1\\1\\1\\1\\1]",
		tag::display_location(vfunc, {cfunc}, treg).c_str());

}


#endif // DISABLE_LOCATOR_TEST
