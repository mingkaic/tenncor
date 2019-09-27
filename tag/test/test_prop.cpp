
#ifndef DISABLE_PROP_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "tag/test/common.hpp"

#include "tag/prop.hpp"


TEST(PROP, Tag)
{
	tag::TagRegistry treg;
	tag::PropertyRegistry registry(treg);
	teq::iTensor* ptr;
	{
		teq::TensptrT tens = std::make_shared<MockTensor>();
		registry.property_tag(tens, "property4");
		registry.property_tag(tens, "property1");
		registry.property_tag(tens, "property2");

		ptr = tens.get();

		EXPECT_FALSE(registry.has_property(ptr, "property3"));
		EXPECT_FALSE(registry.has_property(ptr, "property0"));

		EXPECT_TRUE(registry.has_property(ptr, "property1"));
		EXPECT_TRUE(registry.has_property(ptr, "property2"));
		EXPECT_TRUE(registry.has_property(ptr, "property4"));
	}

	EXPECT_FALSE(registry.has_property(ptr, "property3"));
	EXPECT_FALSE(registry.has_property(ptr, "property0"));
	EXPECT_FALSE(registry.has_property(ptr, "property1"));
	EXPECT_FALSE(registry.has_property(ptr, "property2"));
	EXPECT_FALSE(registry.has_property(ptr, "property4"));
}


#endif // DISABLE_PROP_TEST
