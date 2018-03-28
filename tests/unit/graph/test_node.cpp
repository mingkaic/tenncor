//
// Created by Mingkai Chen on 2017-03-10.
//

#ifndef DISABLE_GRAPH_MODULE_TESTS

#include "gtest/gtest.h"

#include "fuzz/fuzz.hpp"
#include "check.hpp"
#include "print.hpp"

#include "graph/graph.hpp"
#include "mock_node.hpp"


#ifndef DISABLE_NODE_TEST


class NODE : public testify::fuzz_test
{
protected:
	virtual void SetUp (void) {}

	virtual void TearDown (void)
	{
		testify::fuzz_test::TearDown();
		testify::mocker::clear();
	}
};


using namespace testutils;


// covers inode: constructors and graph
TEST_F(NODE, OnGraph_B000)
{
	std::string label = get_string(get_int(1, "label.size", {14, 29})[0], "label");
	mock_node* ptr;
	{
		mock_node node(label);
		ptr = &node;
		EXPECT_TRUE(nnet::graph::get_global().has_node(&node)) <<
			sprintf("node %s is not registered in graph", node.get_uid().c_str());
	}
	EXPECT_FALSE(nnet::graph::get_global().has_node(ptr)) <<
		sprintf("deleted node with ptr %d is not removed from graph", ptr);
}


// covers inode: clone
TEST_F(NODE, Copy_B001)
{
	mock_node assign("");

	std::string label = get_string(get_int(1, "label.size", {14, 29})[0], "label");
	mock_node n1(label);

	mock_node cpy(n1);
	EXPECT_TRUE(nnet::graph::get_global().has_node(&cpy)) <<
		sprintf("copied node %s is not registered in graph", cpy.get_uid().c_str());
	assign = n1;

	EXPECT_EQ(label, n1.get_label());
	EXPECT_EQ(n1.get_label(), cpy.get_label());
	EXPECT_EQ(n1.get_label(), assign.get_label());

	EXPECT_NE(n1.get_uid(), cpy.get_uid());
	EXPECT_NE(n1.get_uid(), assign.get_uid());
}


// covers inode: move
TEST_F(NODE, Move_B001)
{
	mock_node assign("");

	std::string label = get_string(get_int(1, "label.size", {14, 29})[0], "label");
	mock_node n1(label);

	std::string ouid = n1.get_uid();
	EXPECT_EQ(label, n1.get_label());

	mock_node mv(std::move(n1));
	EXPECT_TRUE(nnet::graph::get_global().has_node(&mv)) <<
		sprintf("moved node %s is not registered in graph", mv.get_uid().c_str());
	EXPECT_TRUE(nnet::graph::get_global().has_node(&n1)) <<
		sprintf("original node %s is not registered in graph", n1.get_uid().c_str());

	std::string n1str = n1.get_label();
	std::string ouid2 = mv.get_uid();
	EXPECT_TRUE(n1str.empty()) <<
		sprintf("empty n1 node got label %s", n1str.c_str());
	EXPECT_EQ(label, mv.get_label());
	EXPECT_NE(ouid, ouid2);

	assign = std::move(mv);

	std::string mvstr = mv.get_label();
	std::string ouid3 = assign.get_uid();
	EXPECT_TRUE(mvstr.empty()) <<
		sprintf("empty mv node got label %s", mvstr.c_str());
	EXPECT_EQ(label, assign.get_label());
	EXPECT_NE(ouid, ouid3);
	EXPECT_NE(ouid2, ouid3);
}


// covers inode: get_uid
TEST_F(NODE, UID_B002)
{
	std::unordered_set<std::string> us;
	size_t ns = get_int(1, "ns", {1412, 2922})[0];
	for (size_t i = 0; i < ns; i++)
	{
		mock_node mn("");
		std::string uid = mn.get_uid();
		EXPECT_TRUE(us.end() == us.find(uid)) <<
			sprintf("found duplicate uid \"%s\"", uid.c_str());
		us.emplace(uid);
	}
}


// covers inode: get_label, get_name
TEST_F(NODE, Label_B003)
{
	std::string label = get_string(get_int(1, "label.size", {14, 29})[0], "label");
	mock_node n1(label);

	std::string uid = n1.get_uid();
	EXPECT_EQ(n1.get_name(), "<"+label+":"+uid+">");
	EXPECT_EQ(label, n1.get_label());
}


#endif /* DISABLE_NODE_TEST */


#endif /* DISABLE_GRAPH_MODULE_TESTS */
