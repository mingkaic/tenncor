
#ifndef DISABLE_TAG_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "teq/mock/leaf.hpp"

#include "tag/tag.hpp"

#include "tag/mock/tag.hpp"


TEST(TAG, AddGet)
{
	tag::TagRegistry registry;
	teq::iTensor* ptr;
	teq::TensrefT ref;
	{
		teq::TensptrT tens = std::make_shared<MockTensor>();
		registry.add_tag(tens, std::make_unique<MockTag>());
		EXPECT_EQ(1, registry.registry_.size());
		tag::TagRepsT reps = registry.get_tags(tens.get());
		ASSERT_HAS(reps, tag_key);
		std::vector<std::string> expected_tag_values = tag_values;
		expected_tag_values.push_back("1");
		EXPECT_VECEQ(expected_tag_values, reps[tag_key]);

		auto it = registry.registry_.find(tag::TensKey(tens.get()));
		EXPECT_FALSE(it->first.expired());

		registry.add_tag(tens, std::make_unique<MockTag>());
		EXPECT_EQ(1, registry.registry_.size());
		expected_tag_values.back() = "2";
		reps = registry.get_tags(tens.get());
		EXPECT_VECEQ(expected_tag_values, reps[tag_key]);

		ptr = tens.get();
		ref = tens;
	}
	EXPECT_EQ(1, registry.registry_.size());
	tag::TagRepsT reps = registry.get_tags(ptr);
	EXPECT_EQ(0, reps.size());

	auto it = registry.registry_.find(tag::TensKey(ptr));
	EXPECT_TRUE(it->first.expired());

	EXPECT_FATAL(registry.add_tag(ref, std::make_unique<MockTag>()),
		"cannot tag with expired tensor ref");
}


TEST(TAG, AddMove)
{
	tag::TagRegistry registry;
	teq::TensrefT ref;
	teq::iTensor* ptr;
	teq::iTensor* ptr2;
	{
		teq::TensptrT tens = std::make_shared<MockTensor>();
		{
			teq::TensptrT tens2 = std::make_shared<MockTensor>();

			// move non tagged tens to non tagged tens
			registry.move_tags(tens2, tens.get());
			EXPECT_HASNOT(registry.registry_, tag::TensKey(tens.get()));
			EXPECT_HASNOT(registry.registry_, tag::TensKey(tens2.get()));
			EXPECT_EQ(0, registry.registry_.size());

			registry.add_tag(tens, std::make_unique<MockTag>());
			auto it = registry.registry_.find(tag::TensKey(tens.get()));
			EXPECT_FALSE(it->first.expired());

			// move tagged tens to non tagged tens
			registry.move_tags(tens2, tens.get());
			EXPECT_HASNOT(registry.registry_, tag::TensKey(tens.get()));
			ASSERT_HAS(registry.registry_, tag::TensKey(tens2.get()));
			tag::TagRepsT reps = registry.get_tags(tens2.get());

			ASSERT_HAS(reps, tag_key);
			std::vector<std::string> expected_tag_values = tag_values;
			expected_tag_values.push_back("1");
			EXPECT_VECEQ(expected_tag_values, reps[tag_key]);

			registry.add_tag(tens, std::make_unique<MockTag>());
			it = registry.registry_.find(tag::TensKey(tens.get()));
			EXPECT_FALSE(it->first.expired());

			// move tagged tens to another tagged tens
			registry.move_tags(tens2, tens.get());
			EXPECT_HASNOT(registry.registry_, tag::TensKey(tens.get()));
			ASSERT_HAS(registry.registry_, tag::TensKey(tens2.get()));
			reps = registry.get_tags(tens2.get());

			ASSERT_HAS(reps, tag_key);
			expected_tag_values.back() = "2";
			EXPECT_VECEQ(expected_tag_values, reps[tag_key]);

			std::string second_key = tag_key + "2";
			registry.add_tag(tens, std::make_unique<MockTag>(tid + 1, second_key));
			registry.move_tags(tens2, tens.get());
			EXPECT_HASNOT(registry.registry_, tag::TensKey(tens.get()));
			ASSERT_HAS(registry.registry_, tag::TensKey(tens2.get()));
			reps = registry.get_tags(tens2.get());

			ASSERT_HAS(reps, tag_key);
			expected_tag_values.back() = "2";
			EXPECT_VECEQ(expected_tag_values, reps[tag_key]);

			ASSERT_HAS(reps, second_key);
			expected_tag_values.back() = "1";
			EXPECT_VECEQ(expected_tag_values, reps[second_key]);

			ptr2 = tens2.get();
		}
		ref = tens;
		ptr = tens.get();
		// expect no changes
		registry.move_tags(ref, ptr2);
		EXPECT_HASNOT(registry.registry_, tag::TensKey(ptr));
		ASSERT_HAS(registry.registry_, tag::TensKey(ptr2));
	}
	// expect no changes
	EXPECT_FATAL(registry.move_tags(ref, ptr2),
		"cannot move with expired destination tensor");
}


TEST(TAG, RegistryRetag)
{
	tag::TagRegistry registry;

	teq::iTensor* orig = new MockTensor();
	teq::iTensor* repl = new MockTensor();

	teq::TensptrT tens(orig, [](teq::iTensor* tens){});
	registry.add_tag(tens, std::make_unique<MockTag>(tid + 1, tag_key + "1"));

	auto reps = registry.get_tags(orig);
	EXPECT_EQ(1, reps.size());
	ASSERT_HAS(reps, tag_key + "1");
	std::vector<std::string> expected_tag_values = tag_values;
	expected_tag_values.push_back("1");
	EXPECT_VECEQ(expected_tag_values, reps[tag_key + "1"]);

	tens.reset(repl);

	reps = registry.get_tags(orig);
	EXPECT_EQ(0, reps.size());

	teq::TensptrT retens(orig);
	registry.add_tag(retens, std::make_unique<MockTag>(tid + 1, tag_key + "2"));

	reps = registry.get_tags(orig);
	EXPECT_EQ(1, reps.size());
	ASSERT_HAS(reps, tag_key + "2");
	EXPECT_VECEQ(expected_tag_values, reps[tag_key + "2"]);
}


#endif
