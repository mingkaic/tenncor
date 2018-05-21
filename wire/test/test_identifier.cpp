#ifndef DISABLE_WIRE_MODULE_TESTS

#include "gtest/gtest.h"

#include "fuzzutil/fuzz.hpp"
#include "fuzzutil/sgen.hpp"
#include "fuzzutil/check.hpp"

#include "mold/variable.hpp"

#include "wire/identifier.hpp"


#ifndef DISABLE_IDENTITY_TEST


using namespace testutil;


class IDENTITY : public fuzz_test {};


struct mock_identifier : public wire::Identifier
{
	mock_identifier (wire::Graph* graph, mold::iNode* arg, std::string label) :
        wire::Identifier(graph, arg, label) {}

    mold::iNode* get_node (void) { return wire::Identifier::get_node(); }
};


TEST_F(IDENTITY, Copy_)
{
    wire::Graph& graph = wire::Graph::get_global();
	std::string label = get_string(16, "label");

    mold::Variable* assvar = new mold::Variable();
    mold::Variable* convar = new mold::Variable();
	mock_identifier assign(&graph, assvar, "bad_sample");
	mock_identifier id(&graph, convar, label);
    ASSERT_TRUE(graph.has_node(id.get_uid())) << "id not found in global";

	mock_identifier cp(id);
    EXPECT_TRUE(graph.has_node(cp.get_uid())) << "cp not found in global";
    EXPECT_STRNE(id.get_uid().c_str(), cp.get_uid().c_str());
    EXPECT_STREQ(id.get_label().c_str(), cp.get_label().c_str());
    EXPECT_NE(id.get_node(), cp.get_node());

	assign = id;
    ASSERT_TRUE(graph.has_node(assign.get_uid())) << "assign not found in global";
    EXPECT_STRNE(id.get_uid().c_str(), assign.get_uid().c_str());
    EXPECT_STREQ(id.get_label().c_str(), assign.get_label().c_str());
    EXPECT_NE(id.get_node(), assign.get_node());
}


TEST_F(IDENTITY, Move_)
{

}


#endif /* DISABLE_IDENTITY_TEST */


#endif /* DISABLE_WIRE_MODULE_TESTS */
