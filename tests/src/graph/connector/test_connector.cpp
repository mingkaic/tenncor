//
// Created by Mingkai Chen on 2017-03-14.
//

#ifndef DISABLE_CONNECTOR_MODULE_TESTS

#include <algorithm>

#include "gtest/gtest.h"

#include "tests/include/mocks/mock_node.h"
#include "tests/include/mocks/mock_connector.h"
#include "tests/include/fuzz.h"


#ifndef DISABLE_CONNECTOR_TEST


class CONNECTOR : public FUZZ::fuzz_test {};


// covers iconnector
// copy constructor and assignment
TEST_F(CONNECTOR, Copy_H000)
{
	mocker::usage_.clear();
	mock_connector* assign = new mock_connector(std::vector<inode*>{}, "");
	mock_connector* assign2 = new mock_connector(std::vector<inode*>{}, "");
	mock_connector* assign3 = new mock_connector(std::vector<inode*>{}, "");
	mock_connector* assign4 = new mock_connector(std::vector<inode*>{}, "");

	std::string conname = get_string(get_int(1, "conname.size", {14, 29})[0], "conname");
	std::string conname2 = get_string(get_int(1, "conname2.size", {14, 29})[0], "conname2");
	std::string bossname = get_string(get_int(1, "bossname.size", {14, 29})[0], "bossname");
	std::string bossname2 = get_string(get_int(1, "bossname2.size", {14, 29})[0], "bossname2");
	std::string label1 = get_string(get_int(1, "label1.size", {14, 29})[0], "label1");
	mock_node* n1 = new mock_node(label1);

	mock_connector* conn = new mock_connector(std::vector<inode *>{n1}, conname);
	mock_connector* conn2 = new mock_connector(std::vector<inode *>{n1, n1}, conname2);
	void* ogid1 = conn->get_gid();
	void* ogid2 = conn2->get_gid();

	mock_connector* cpy = new mock_connector(*conn);
	mock_connector* cpy2 = new mock_connector(*conn2);
	void* gid1 = cpy->get_gid();
	void* gid2 = cpy2->get_gid();
	EXPECT_EQ(ogid1, gid1);
	EXPECT_EQ(ogid2, gid2);

	*assign = *conn;
	*assign2 = *conn2;
	void* gid5 = assign->get_gid();
	void* gid6 = assign2->get_gid();
	EXPECT_EQ(ogid1, gid5);
	EXPECT_EQ(ogid2, gid6);

	// test base connectors before testing connectors of connectors
	// to prevent optimizations from deleting gids thereby allowing duplicate gid addresses.
	// (without allocator randomization)

	mock_connector* boss = new mock_connector(std::vector<inode *>{conn, n1}, bossname);
	mock_connector* boss2 = new mock_connector(std::vector<inode *>{conn, conn2}, bossname2);
	void* ogid3 = boss->get_gid();
	void* ogid4 = boss2->get_gid();

	mock_connector* cpy3 = new mock_connector(*boss);
	mock_connector* cpy4 = new mock_connector(*boss2);
	void* gid3 = cpy3->get_gid();
	void* gid4 = cpy4->get_gid();
	EXPECT_EQ(ogid3, gid3);
	EXPECT_EQ(ogid4, gid4);

	*assign3 = *boss;
	*assign4 = *boss2;
	void* gid7 = assign3->get_gid();
	void* gid8 = assign4->get_gid();
	EXPECT_EQ(ogid3, gid7);
	EXPECT_EQ(ogid4, gid8);

	conn->inst_ = "conn";
	conn2->inst_ = "conn2";
	boss->inst_ = "boss";
	boss2->inst_ = "boss2";
	cpy->inst_ = "cpy";
	cpy2->inst_ = "cpy2";
	cpy3->inst_ = "cpy3";
	cpy4->inst_ = "cpy4";
	assign->inst_ = "assign";
	assign2->inst_ = "assign2";
	assign3->inst_ = "assign3";
	assign4->inst_ = "assign4";

	delete n1;
	delete conn;
	delete conn2;
	delete boss;
	delete boss2;
	delete assign;
	delete assign2;
	delete assign3;
	delete assign4;
	delete cpy;
	delete cpy2;
	delete cpy3;
	delete cpy4;
	EXPECT_TRUE(mocker::EXPECT_CALL("conn::death_on_broken", 1));
	EXPECT_TRUE(mocker::EXPECT_CALL("conn2::death_on_broken", 1));
	EXPECT_TRUE(mocker::EXPECT_CALL("boss::death_on_broken", 2));
	EXPECT_TRUE(mocker::EXPECT_CALL("boss2::death_on_broken", 2));
	EXPECT_TRUE(mocker::EXPECT_CALL("cpy::death_on_broken", 1));
	EXPECT_TRUE(mocker::EXPECT_CALL("cpy2::death_on_broken", 1));
	EXPECT_TRUE(mocker::EXPECT_CALL("cpy3::death_on_broken", 2));
	EXPECT_TRUE(mocker::EXPECT_CALL("cpy4::death_on_broken", 2));
	EXPECT_TRUE(mocker::EXPECT_CALL("assign::death_on_broken", 1));
	EXPECT_TRUE(mocker::EXPECT_CALL("assign2::death_on_broken", 1));
	EXPECT_TRUE(mocker::EXPECT_CALL("assign3::death_on_broken", 2));
	EXPECT_TRUE(mocker::EXPECT_CALL("assign4::death_on_broken", 2));
}


// covers iconnector
// move constructor and assignment
TEST_F(CONNECTOR, Move_H000)
{
	mocker::usage_.clear();
	mock_connector* assign = new mock_connector(std::vector<inode*>{}, "");
	mock_connector* assign2 = new mock_connector(std::vector<inode*>{}, "");
	mock_connector* assign3 = new mock_connector(std::vector<inode*>{}, "");
	mock_connector* assign4 = new mock_connector(std::vector<inode*>{}, "");

	std::string conname = get_string(get_int(1, "conname.size", {14, 29})[0], "conname");
	std::string conname2 = get_string(get_int(1, "conname2.size", {14, 29})[0], "conname2");
	std::string bossname = get_string(get_int(1, "bossname.size", {14, 29})[0], "bossname");
	std::string bossname2 = get_string(get_int(1,"bossname2.size",  {14, 29})[0], "bossname2");
	std::string label1 = get_string(get_int(1, "label1.size", {14, 29})[0], "label1");
	mock_node* n1 = new mock_node(label1);

	mock_connector* conn = new mock_connector(std::vector<inode *>{n1}, conname);
	mock_connector* conn2 = new mock_connector(std::vector<inode *>{n1, n1}, conname2);
	void* ogid1 = conn->get_gid();
	void* ogid2 = conn2->get_gid();

	mock_connector* mv = new mock_connector(std::move(*conn));
	mock_connector* mv2 = new mock_connector(std::move(*conn2));
	EXPECT_EQ(nullptr, conn->get_gid());
	EXPECT_EQ(nullptr, conn2->get_gid());
	EXPECT_EQ(ogid1, mv->get_gid());
	EXPECT_EQ(ogid2, mv2->get_gid());

	*assign = std::move(*mv);
	*assign2 = std::move(*mv2);
	EXPECT_EQ(nullptr, mv->get_gid());
	EXPECT_EQ(nullptr, mv2->get_gid());
	EXPECT_EQ(ogid1, assign->get_gid());
	EXPECT_EQ(ogid2, assign2->get_gid());

	// test base connectors before testing connectors of connectors
	// to prevent optimizations from deleting gids thereby allowing duplicate gid addresses.
	// (without allocator randomization)

	mock_connector* boss = new mock_connector(std::vector<inode *>{assign, n1}, bossname);
	mock_connector* boss2 = new mock_connector(std::vector<inode *>{assign, assign2}, bossname2);
	void* ogid3 = boss->get_gid();
	void* ogid4 = boss2->get_gid();

	mock_connector* mv3 = new mock_connector(std::move(*boss));
	mock_connector* mv4 = new mock_connector(std::move(*boss2));
	EXPECT_EQ(nullptr, boss->get_gid());
	EXPECT_EQ(nullptr, boss2->get_gid());
	EXPECT_EQ(ogid3, mv3->get_gid());
	EXPECT_EQ(ogid4, mv4->get_gid());

	*assign3 = std::move(*mv3);
	*assign4 = std::move(*mv4);
	EXPECT_EQ(nullptr, mv3->get_gid());
	EXPECT_EQ(nullptr, mv4->get_gid());
	EXPECT_EQ(ogid3, assign3->get_gid());
	EXPECT_EQ(ogid4, assign4->get_gid());

	conn->inst_ = "conn";
	conn2->inst_ = "conn2";
	boss->inst_ = "boss";
	boss2->inst_ = "boss2";
	mv->inst_ = "cpy";
	mv2->inst_ = "cpy2";
	mv3->inst_ = "cpy3";
	mv4->inst_ = "cpy4";
	assign->inst_ = "assign";
	assign2->inst_ = "assign2";
	assign3->inst_ = "assign3";
	assign4->inst_ = "assign4";

	delete n1;
	delete conn;
	delete conn2;
	delete boss;
	delete boss2;
	delete assign;
	delete assign2;
	delete assign3;
	delete assign4;
	delete mv;
	delete mv2;
	delete mv3;
	delete mv4;
	EXPECT_TRUE(mocker::EXPECT_CALL("conn::death_on_broken", 0));
	EXPECT_TRUE(mocker::EXPECT_CALL("conn2::death_on_broken", 0));
	EXPECT_TRUE(mocker::EXPECT_CALL("boss::death_on_broken", 0));
	EXPECT_TRUE(mocker::EXPECT_CALL("boss2::death_on_broken", 0));
	EXPECT_TRUE(mocker::EXPECT_CALL("cpy::death_on_broken", 0));
	EXPECT_TRUE(mocker::EXPECT_CALL("cpy2::death_on_broken", 0));
	EXPECT_TRUE(mocker::EXPECT_CALL("cpy3::death_on_broken", 0));
	EXPECT_TRUE(mocker::EXPECT_CALL("cpy4::death_on_broken", 0));
	EXPECT_TRUE(mocker::EXPECT_CALL("assign::death_on_broken", 1));
	EXPECT_TRUE(mocker::EXPECT_CALL("assign2::death_on_broken", 1));
	EXPECT_TRUE(mocker::EXPECT_CALL("assign3::death_on_broken", 2));
	EXPECT_TRUE(mocker::EXPECT_CALL("assign4::death_on_broken", 2));
}


// covers iconnector
// get_name
TEST_F(CONNECTOR, Name_H001)
{
	mocker::usage_.clear();
	size_t nargs = get_int(1, "nargs", {2, 7})[0];
	std::vector<inode*> ns;
	std::vector<size_t> nlens = get_int(nargs, "nlens", {14, 29});
	std::string argname = "";
	for (size_t i = 0; i < nargs; i++)
	{
		std::string label = get_string(nlens[0], "label");
		argname += label + ",";
		ns.push_back(new mock_node(label));
	};
	argname.pop_back(); // remove last comma
	std::string bossname = get_string(get_int(1, "bossname.size", {14, 29})[0], "bossname");
	mock_connector* conn1 = new mock_connector(ns, bossname);
	std::string expectname = "<" + bossname + ":" + boost::uuids::to_string(conn1->get_uid()) + ">(" + argname + ")";
	EXPECT_EQ(expectname, conn1->get_name());

	conn1->inst_ = "conn1";
	for (inode* n : ns)
	{
		delete n;
	}
	delete conn1;
	EXPECT_TRUE(mocker::EXPECT_CALL("conn1::death_on_broken", nargs));
}


// covers iconnector
// update_graph, is_same_graph
TEST_F(CONNECTOR, Graph_H002)
{
	mocker::usage_.clear();
	std::string conname = get_string(get_int(1, "conname.size", {14, 29})[0], "conname");
	std::string conname2 = get_string(get_int(1, "conname2.size", {14, 29})[0], "conname2");
	std::string bossname = get_string(get_int(1, "bossname.size", {14, 29})[0], "bossname");
	std::string bossname2 = get_string(get_int(1, "bossname2.size", {14, 29})[0], "bossname2");
	std::string label1 = get_string(get_int(1, "label1.size", {14, 29})[0], "label1");
	mock_node* n1 = new mock_node(label1);
	mock_connector* conn = new mock_connector(std::vector<inode *>{n1}, conname);
	mock_connector* conn2 = new mock_connector(std::vector<inode *>{n1, n1}, conname2);

	EXPECT_FALSE(conn->is_same_graph(conn2));
	mock_connector* boss = new mock_connector(std::vector<inode *>{conn, n1}, bossname);
	EXPECT_FALSE(conn2->is_same_graph(boss));
	mock_connector* boss2 = new mock_connector(std::vector<inode *>{conn, conn2}, bossname2);

	// boss2 connects conn and conn2
	EXPECT_TRUE(conn->is_same_graph(conn2));
	EXPECT_TRUE(conn2->is_same_graph(boss));

	EXPECT_TRUE(boss->is_same_graph(conn));
	EXPECT_TRUE(boss2->is_same_graph(boss2));

	conn->inst_ = "conn";
	conn2->inst_ = "conn2";
	boss->inst_ = "boss";
	boss2->inst_ = "boss2";
	delete n1;
	delete conn;
	delete conn2;
	delete boss;
	delete boss2;
	EXPECT_TRUE(mocker::EXPECT_CALL("conn::death_on_broken", 1));
	EXPECT_TRUE(mocker::EXPECT_CALL("conn2::death_on_broken", 1));
	EXPECT_TRUE(mocker::EXPECT_CALL("boss::death_on_broken", 2));
	EXPECT_TRUE(mocker::EXPECT_CALL("boss2::death_on_broken", 2));
}


// covers iconnector
// potential_descendent
TEST_F(CONNECTOR, Descendent_H003)
{
	mocker::usage_.clear();
	std::string conname = get_string(get_int(1, "conname.size", {14, 29})[0], "conname");
	std::string conname2 = get_string(get_int(1, "conname2.size", {14, 29})[0], "conname2");
	std::string bossname = get_string(get_int(1, "bossname.size", {14, 29})[0], "bossname");
	std::string bossname2 = get_string(get_int(1, "bossname2.size", {14, 29})[0], "bossname2");
	std::string label1 = get_string(get_int(1, "label1.size", {14, 29})[0], "label1");
	std::string label2 = get_string(get_int(1, "label2.size", {14, 29})[0], "label2");
	std::string label3 = get_string(get_int(1, "label3.size", {14, 29})[0], "label3");
	mock_node* n1 = new mock_node(label1);
	mock_node* n2 = new mock_node(label2);
	mock_node* n3 = new mock_node(label3);
	mock_connector* conn = new mock_connector(std::vector<inode *>{n1}, conname);
	mock_connector* conn2 = new mock_connector(std::vector<inode *>{n1, n1}, conname2);
	mock_connector* separate = new mock_connector(std::vector<inode*>{n3, n2}, conname2);
	mock_connector* boss = new mock_connector(std::vector<inode *>{n1, n2}, conname2);

	conn->inst_ = "conn";
	conn2->inst_ = "conn2";
	boss->inst_ = "boss";
	separate->inst_ = "separate";
	conn->potential_descendent(conn2);
	conn2->potential_descendent(conn);
	boss->potential_descendent(separate);
	separate->potential_descendent(boss);
	EXPECT_TRUE(mocker::EXPECT_CALL("conn::get_leaves1", 2));
	EXPECT_TRUE(mocker::EXPECT_CALL("conn2::get_leaves1", 2));
	EXPECT_TRUE(mocker::EXPECT_CALL("boss::get_leaves1", 2));
	EXPECT_TRUE(mocker::EXPECT_CALL("separate::get_leaves1", 2));

	delete n1;
	delete n2;
	delete n3;
	delete conn;
	delete conn2;
	delete boss;
	delete separate;
	EXPECT_TRUE(mocker::EXPECT_CALL("conn::death_on_broken", 1));
	EXPECT_TRUE(mocker::EXPECT_CALL("conn2::death_on_broken", 1));
	EXPECT_TRUE(mocker::EXPECT_CALL("boss::death_on_broken", 2));
	EXPECT_TRUE(mocker::EXPECT_CALL("separate::death_on_broken", 2));
}


#endif /* DISABLE_CONNECTOR_TEST */


#endif /* DISABLE_CONNECTOR_MODULE_TESTS */
