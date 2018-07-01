#ifndef DISABLE_KILN_MODULE_TESTS

#include "gtest/gtest.h"

#include "fuzzutil/fuzz.hpp"
#include "fuzzutil/sgen.hpp"
#include "fuzzutil/check.hpp"

#include "mold/variable.hpp"

#include "kiln/identifier.hpp"


#ifndef DISABLE_IDENTITY_TEST


using namespace testutil;


class IDENTIFIER : public fuzz_test
{
protected:
	virtual void SetUp (void) {}

	virtual void TearDown (void)
	{
		testutil::fuzz_test::TearDown();
		kiln::Graph& g = kiln::Graph::get_global();
		assert(0 == g.size());
	}
};


struct mock_identifier : public kiln::Identifier
{
	mock_identifier (kiln::Graph* graph, mold::iNode* arg, std::string label) :
		kiln::Identifier(graph, arg, label) {}

	kiln::Graph* get_graph (void) const { return graph_; }

	mold::iNode* get_node (void) const
	{
		mold::iNode* out = nullptr;
		if (nullptr != death_sink_)
		{
			out = get();
		}
		return out;
	}
};


TEST_F(IDENTIFIER, Copy_C000)
{
	kiln::Graph& graph = kiln::Graph::get_global();
	std::string label = get_string(16, "label");

	mold::Variable* assvar = new mold::Variable();
	mold::Variable* convar = new mold::Variable();
	mock_identifier assign(&graph, assvar, "bad_sample");
	mock_identifier id(&graph, convar, label);
	ASSERT_TRUE(graph.has_node(id.get_uid())) << "id not found in global";
	ASSERT_EQ(convar, id.get_node());
	ASSERT_EQ(&graph, id.get_graph());

	mock_identifier cp(id);
	EXPECT_TRUE(graph.has_node(cp.get_uid())) << "cp not found in global";
	EXPECT_NE(id.get_uid(), cp.get_uid());
	EXPECT_STREQ(label.c_str(), cp.get_label().c_str());
	EXPECT_NE(convar, cp.get_node());

	assign = id;
	ASSERT_TRUE(graph.has_node(assign.get_uid())) << "assign not found in global";
	EXPECT_NE(id.get_uid(), assign.get_uid());
	EXPECT_STREQ(label.c_str(), assign.get_label().c_str());
	EXPECT_NE(convar, assign.get_node());
}


TEST_F(IDENTIFIER, Move_C001)
{
	kiln::Graph& graph = kiln::Graph::get_global();
	std::string label = get_string(16, "label");

	mold::Variable* assvar = new mold::Variable();
	mold::Variable* convar = new mold::Variable();
	mock_identifier assign(&graph, assvar, "bad_sample");
	mock_identifier id(&graph, convar, label);
	ASSERT_TRUE(graph.has_node(id.get_uid())) << "id not found in global";
	ASSERT_EQ(convar, id.get_node());
	ASSERT_EQ(&graph, id.get_graph());

	mock_identifier mv(std::move(id));
	EXPECT_TRUE(graph.has_node(mv.get_uid())) << "mv not found in global";
	EXPECT_NE(id.get_uid(), mv.get_uid());
	EXPECT_STREQ(label.c_str(), mv.get_label().c_str());
	EXPECT_EQ(convar, mv.get_node());
	EXPECT_EQ(&graph, mv.get_graph());

	EXPECT_FALSE(graph.has_node(id.get_uid()));
	EXPECT_EQ(nullptr, id.get_node());
	EXPECT_EQ(nullptr, id.get_graph());

	assign = std::move(mv);
	ASSERT_TRUE(graph.has_node(assign.get_uid())) << "assign not found in global";
	EXPECT_NE(id.get_uid(), assign.get_uid());
	EXPECT_STREQ(label.c_str(), assign.get_label().c_str());
	EXPECT_EQ(convar, assign.get_node());
	EXPECT_EQ(&graph, assign.get_graph());

	EXPECT_FALSE(graph.has_node(mv.get_uid()));
	EXPECT_EQ(nullptr, mv.get_node());
	EXPECT_EQ(nullptr, mv.get_graph());
}


#endif /* DISABLE_IDENTITY_TEST */


#endif /* DISABLE_KILN_MODULE_TESTS */