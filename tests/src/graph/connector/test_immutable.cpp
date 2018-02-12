//
// Created by Mingkai Chen on 2017-03-14.
//

#ifndef DISABLE_CONNECTOR_MODULE_TESTS

#include <algorithm>

#include "gtest/gtest.h"

#include "tests/include/mocks/mock_elem_op.h"
#include "tests/include/mocks/mock_node.h"
#include "tests/include/mocks/mock_itensor.h"

#include "include/graph/leaf/variable.hpp"


#ifndef DISABLE_IMMUTABLE_TEST


class IMMUTABLE : public FUZZ::fuzz_test {};


#ifdef SLOW_GRAPH
static std::pair<size_t,size_t> nnodes_range = {131, 297};
#elif defined(THOROUGH_GRAPH)
static std::pair<size_t,size_t> nnodes_range = {67, 131};
#elif defined(FAST_GRAPH)
static std::pair<size_t,size_t> nnodes_range = {31, 67};
#else // FASTEST_GRAPH
static std::pair<size_t,size_t> nnodes_range = {17, 31};
#endif


struct cond_actor : public tens_template<double>
{
	cond_actor (out_wrapper<void> dest,
		std::vector<in_wrapper<void> > srcs, bool& mutate) :
	tens_template<double>(dest, srcs), mutate_(&mutate) {}

	virtual void action (void)
	{
		size_t n_elems = this->dest_.second.n_elems();
		for (size_t i = 0; i < this->srcs_.size(); i++)
		{
			n_elems = std::min(n_elems, this->srcs_[i].second.n_elems());
		}
		for (size_t j = 0; j < n_elems; j++)
		{
			this->dest_.first[j] = 0;
			if (*mutate_)
			{
				this->dest_.first[j] = this->srcs_.size();
			}
			for (size_t i = 0; i < this->srcs_.size(); i++)
			{
				this->dest_.first[j] += this->srcs_[i].first[j];
			}
		}
	}

private:
	bool* mutate_;
};


static bool bottom_up (std::vector<iconnector*> ordering)
{
	// ordering travels from leaf towards the root
	// ordering test ensures get leaf is a bottom-up procedure
	// todo: some how test caching performance (probabilistically increase hit rate as i increases)
	// eventually most nodes in traversals should be cached, so ordering size should decrease
	bool o = true;
	iconnector* last = nullptr;
	for (iconnector* ord : ordering)
	{
		if (last)
		{
			// ord should be parent of last
			o = o && ord->has_subject(last);
		}
		last = ord;
	}
	return o;
}


TEST_F(IMMUTABLE, Copy_I000)
{
	elem_op* assign = new mock_elem_op(std::vector<inode*>{}, "", get_testshaper(this));
	elem_op* central = new mock_elem_op(std::vector<inode*>{}, "", get_testshaper(this));
	const tensor* res = central->eval();

	elem_op* cpy = central->clone();
	*assign = *central;
	ASSERT_NE(nullptr, cpy);

	const tensor_double* cres = dynamic_cast<const tensor_double*>(cpy->eval());
	const tensor_double* ares = dynamic_cast<const tensor_double*>(assign->eval());

	std::vector<double> data = expose<double>(central);
	std::vector<double> cdata = expose<double>(cpy);
	std::vector<double> adata = expose<double>(assign);

	EXPECT_TRUE(tensorshape_equal(res->get_shape(), cres->get_shape()));
	EXPECT_TRUE(tensorshape_equal(res->get_shape(), ares->get_shape()));
	EXPECT_TRUE(std::equal(data.begin(), data.end(), cdata.begin()));
	EXPECT_TRUE(std::equal(data.begin(), data.end(), adata.begin()));

	delete assign;
	delete cpy;
	delete central;
}


TEST_F(IMMUTABLE, Move_I000)
{
	elem_op* assign  = new mock_elem_op(std::vector<inode*>{}, "", get_testshaper(this));
	elem_op* central = new mock_elem_op(std::vector<inode*>{}, "", get_testshaper(this));
	const tensor_double* res = dynamic_cast<const tensor_double*>(central->eval());
	std::vector<double> data = expose<double>(central);
	tensorshape rs = res->get_shape();

	elem_op* mv = central->move();
	EXPECT_NE(nullptr, mv);

	const tensor_double* mres = dynamic_cast<const tensor_double*>(mv->eval());
	std::vector<double> mdata = expose<double>(mv);
	tensorshape ms = mres->get_shape();

	EXPECT_EQ(nullptr, central->eval());

	*assign = std::move(*mv);
	const tensor_double* ares = dynamic_cast<const tensor_double*>(assign->eval());
	std::vector<double> adata = expose<double>(assign);
	tensorshape as = ares->get_shape();

	EXPECT_EQ(nullptr, mv->eval());

	EXPECT_TRUE(tensorshape_equal(rs, ms));
	EXPECT_TRUE(tensorshape_equal(rs, as));
	EXPECT_TRUE(std::equal(data.begin(), data.end(), mdata.begin()));
	EXPECT_TRUE(std::equal(data.begin(), data.end(), adata.begin()));

	delete assign;
	delete mv;
	delete central;
}


TEST_F(IMMUTABLE, Descendent_I001)
{
	std::string conname = get_string(get_int(1, "conname.size", {14, 29})[0], "conname");
	std::string conname2 = get_string(get_int(1, "conname2.size", {14, 29})[0], "conname2");
	std::string bossname = get_string(get_int(1, "bossname.size", {14, 29})[0], "bossname");
	std::string bossname2 = get_string(get_int(1, "bossname2.size", {14, 29})[0], "bossname2");
	std::string label1 = get_string(get_int(1, "label1.size", {14, 29})[0], "label1");
	std::string label2 = get_string(get_int(1, "label2.size", {14, 29})[0], "label2");
	std::string label3 = get_string(get_int(1, "label3.size", {14, 29})[0], "label3");
	std::vector<double> leafvalue = get_double(3, "leafvalue");
	variable* n1 = new variable(leafvalue[0], label1);
	variable* n2 = new variable(leafvalue[1], label2);
	variable* n3 = new variable(leafvalue[2], label3);

	elem_op* conn = new mock_elem_op(std::vector<inode *>{n1}, conname, get_testshaper(this));
	elem_op* conn2 = new mock_elem_op(std::vector<inode *>{n1, n1}, conname2, get_testshaper(this));
	elem_op* separate = new mock_elem_op(std::vector<inode*>{n3, n2}, conname2, get_testshaper(this));
	elem_op* boss = new mock_elem_op(std::vector<inode *>{n1, n2}, conname2, get_testshaper(this));

	EXPECT_TRUE(conn->potential_descendent(conn));
	EXPECT_TRUE(conn->potential_descendent(conn2));
	EXPECT_TRUE(conn2->potential_descendent(conn));
	EXPECT_TRUE(conn2->potential_descendent(conn2));
	EXPECT_TRUE(boss->potential_descendent(conn));
	EXPECT_TRUE(boss->potential_descendent(conn2));

	EXPECT_FALSE(separate->potential_descendent(conn));
	EXPECT_FALSE(separate->potential_descendent(conn2));
	EXPECT_FALSE(separate->potential_descendent(boss));
	EXPECT_FALSE(conn->potential_descendent(boss));
	EXPECT_FALSE(conn2->potential_descendent(boss));

	delete conn;
	delete conn2;
	delete boss;
	delete separate;

	delete n1;
	delete n2;
	delete n3;
}


TEST_F(IMMUTABLE, Status_I002)
{
	std::string conname = get_string(get_int(1, "conname.size", {14, 29})[0], "conname");
	std::string conname2 = get_string(get_int(1, "conname2.size", {14, 29})[0], "conname2");
	std::string conname3 = get_string(get_int(1, "conname3.size", {14, 29})[0], "conname3");
	std::string conname4 = get_string(get_int(1, "conname4.size", {14, 29})[0], "conname4");
	std::string conname5 = get_string(get_int(1, "conname5.size", {14, 29})[0], "conname5");
	std::string label1 = get_string(get_int(1, "label1.size", {14, 29})[0], "label1");
	std::string label2 = get_string(get_int(1, "label2.size", {14, 29})[0], "label2");
	std::string label3 = get_string(get_int(1, "label3.size", {14, 29})[0], "label3");
	std::string label4 = get_string(get_int(1, "label4.size", {14, 29})[0], "label4");

	mock_node* n1 = new mock_node(label1);
	mock_node* n2 = new mock_node(label2);
	mock_node* n3 = new mock_node(label3);
	mock_node* n4 = new mock_node(label4);

	tensorshape n1s = random_def_shape(this);
	tensorshape n2s = random_def_shape(this);
	tensorshape n3s = random_def_shape(this);
	n1->data_ = new mock_itensor(this, n1s);
	n2->data_ = new mock_itensor(this, n2s);
	n3->data_ = new mock_itensor(this, n3s);

	elem_op* conn = new mock_elem_op({n1}, conname, get_testshaper(this));
	elem_op* conn2 = new mock_elem_op({n2, n3}, conname2, get_testshaper(this));
	// bad statuses
	elem_op* conn3 = new mock_elem_op({n4, n3}, conname3, get_testshaper(this));
	elem_op* conn4 = new mock_elem_op({n1, n4}, conname4, get_testshaper(this));
	elem_op* conn5 = new mock_elem_op({n2, n4}, conname5, get_testshaper(this));

	EXPECT_TRUE(conn->good_status());
	EXPECT_FALSE(conn2->good_status());
	conn2->eval();
	EXPECT_TRUE(conn2->good_status());
	EXPECT_FALSE(conn3->good_status());
	EXPECT_FALSE(conn4->good_status());
	EXPECT_FALSE(conn5->good_status());

	delete conn;
	delete conn2;
	delete conn3;
	delete conn4;
	delete conn5;
	delete n1;
	delete n2;
	delete n3;
	delete n4;
}


TEST_F(IMMUTABLE, Shape_I003)
{
	std::string conname = get_string(get_int(1, "conname.size", {14, 29})[0], "conname");
	std::string conname2 = get_string(get_int(1, "conname2.size", {14, 29})[0], "conname2");
	std::string conname3 = get_string(get_int(1, "conname3.size", {14, 29})[0], "conname3");
	std::string conname4 = get_string(get_int(1, "conname4.size", {14, 29})[0], "conname4");
	std::string conname5 = get_string(get_int(1, "conname5.size", {14, 29})[0], "conname5");
	std::string label1 = get_string(get_int(1, "label1.size", {14, 29})[0], "label1");
	std::string label2 = get_string(get_int(1, "label2.size", {14, 29})[0], "label2");
	std::string label3 = get_string(get_int(1, "label3.size", {14, 29})[0], "label3");
	std::string label4 = get_string(get_int(1, "label4.size", {14, 29})[0], "label4");

	mock_node* n1 = new mock_node(label1);
	mock_node* n2 = new mock_node(label2);
	mock_node* n3 = new mock_node(label3);
	mock_node* n4 = new mock_node(label4); // status is bad

	// mock tensors initialize with random data...
	tensorshape n1s = random_def_shape(this, 2, 10, 17, 4372);
	tensorshape n2s = random_def_shape(this, 2, 10, 17, 4372);
	tensorshape n3s = random_def_shape(this, 2, 10, 17, 4372);
	n1->data_ = new mock_itensor(this, n1s);
	n2->data_ = new mock_itensor(this, n2s);
	n3->data_ = new mock_itensor(this, n3s);

	// for this test, we only care about shape
	auto fittershaper = [](std::vector<tensorshape> ts) -> tensorshape
	{
		std::vector<size_t> res;
		for (tensorshape& s : ts)
		{
			std::vector<size_t> slist = s.as_list();
			size_t minrank = std::min(slist.size(), res.size());
			for (size_t i = 0; i < minrank; i++)
			{
				res[i] = std::max(res[i], slist[i]);
			}
			if (slist.size()> res.size())
			{
				for (size_t i = minrank; i < slist.size(); i++)
				{
					res.push_back(slist[i]);
				}
			}
		}
		return res;
	};

	elem_op* conn = new mock_elem_op({n1}, conname, fittershaper);
	elem_op* conn2 = new mock_elem_op({n2, n3}, conname2, fittershaper);
	// bad statuses
	elem_op* conn3 = new mock_elem_op({n4, n3}, conname3, fittershaper);
	elem_op* conn4 = new mock_elem_op({n1, n4}, conname4, fittershaper);
	elem_op* conn5 = new mock_elem_op({n2, n4}, conname5, fittershaper);

	// sample expectations
	tensorshape c2shape = fittershaper({n2s, n3s});

	EXPECT_TRUE(tensorshape_equal(n1s, conn->get_shape()));
	EXPECT_TRUE(tensorshape_equal(c2shape, conn2->get_shape()));

	// bad status returns undefined shapes
	EXPECT_FALSE(conn3->get_shape().is_part_defined()); // not part defined is undefined
	EXPECT_FALSE(conn4->get_shape().is_part_defined());
	EXPECT_FALSE(conn5->get_shape().is_part_defined());

	// delete connectors before nodes to avoid triggering suicides
	delete conn;
	delete conn2;
	delete conn3;
	delete conn4;
	delete conn5;
	delete n1;
	delete n2;
	delete n3;
	delete n4;
}


TEST_F(IMMUTABLE, Tensor_I004)
{
	std::string conname = get_string(get_int(1, "conname.size", {14, 29})[0], "conname");
	std::string conname2 = get_string(get_int(1, "conname2.size", {14, 29})[0], "conname2");
	std::string conname3 = get_string(get_int(1, "conname3.size", {14, 29})[0], "conname3");
	std::string conname4 = get_string(get_int(1, "conname4.size", {14, 29})[0], "conname4");
	std::string conname5 = get_string(get_int(1, "conname5.size", {14, 29})[0], "conname5");
	std::string label1 = get_string(get_int(1, "label1.size", {14, 29})[0], "label1");
	std::string label2 = get_string(get_int(1, "label2.size", {14, 29})[0], "label2");
	std::string label3 = get_string(get_int(1, "label3.size", {14, 29})[0], "label3");
	std::string label4 = get_string(get_int(1, "label4.size", {14, 29})[0], "label4");

	mock_node* n1 = new mock_node(label1);
	mock_node* n2 = new mock_node(label2);
	mock_node* n3 = new mock_node(label3);
	mock_node* n4 = new mock_node(label4); // status is bad

	tensorshape n1s = random_def_shape(this);
	tensorshape n2s = random_def_shape(this);
	tensorshape n3s = random_def_shape(this);
	n1->data_ = new mock_itensor(this, n1s);
	n2->data_ = new mock_itensor(this, n2s);
	n3->data_ = new mock_itensor(this, n3s);

	// for this test, we care about data, grab the largest shape, and sum all data that fit in said array
	auto minshaper = [](std::vector<tensorshape> ts)
	{
		tensorshape res = ts[0];
		for (size_t i = 1, n = ts.size(); i < n; i++)
		{
			if (res.n_elems()> ts[i].n_elems())
			{
				res = ts[i];
			}
		}
		return res;
	};

	elem_op* conn = new mock_elem_op(
		{n1}, conname, minshaper, adder);
	elem_op* conn2 = new mock_elem_op(
		{n2, n3}, conname, minshaper, adder);
	// bad statuses
	elem_op* conn3 = new mock_elem_op(
		{n4, n3}, conname3, minshaper, adder);
	elem_op* conn4 = new mock_elem_op(
		{n1, n4}, conname4, minshaper, adder);
	elem_op* conn5 = new mock_elem_op(
		{n2, n4}, conname5, minshaper, adder);

	tensorshape t2 = n2->get_shape();
	tensorshape t3 = n3->get_shape();
	tensorshape c2s = minshaper({t2, t3});
	size_t nc2s = c2s.n_elems();
	std::vector<double> vn2 = expose<double>(n2);
	std::vector<double> vn3 = expose<double>(n3);
	ASSERT_EQ(nc2s, std::min(vn2.size(), vn3.size()));
	ASSERT_EQ(vn2.size(), t2.n_elems());
	ASSERT_EQ(vn3.size(), t3.n_elems());
	double* expectc2 = new double[nc2s];
	{
		std::vector<double> expectin;
		size_t n = std::min(vn2.size(), vn3.size());
		for (size_t i = 0; i < n; i++)
		{
			expectin.push_back(vn2[i]);
			expectin.push_back(vn3[i]);
		}
		for (size_t i = n; i < vn2.size(); i++)
		{
			expectin.push_back(vn2[i]);
			expectin.push_back(0);
		}
		for (size_t i = n; i < vn3.size(); i++)
		{
			expectin.push_back(0);
			expectin.push_back(vn3[i]);
		}

		std::vector<double> v2 = expose<double>(n2);
		std::vector<double> v3 = expose<double>(n3);
		std::vector<in_wrapper<void> > vsinput = {
			in_wrapper<void>{&v2[0], n2s},
			in_wrapper<void>{&v3[0], n3s},
		};
		out_wrapper<void> dest{expectc2, minshaper({n2s, n3s})};
		itens_actor* actor = adder(dest, vsinput, nnet::DOUBLE);
		actor->action();
		delete actor;
	}

	const tensor_double* c1tensor = dynamic_cast<const tensor_double*>(conn->eval());
	const tensor_double* c2tensor = dynamic_cast<const tensor_double*>(conn2->eval());
	ASSERT_NE(nullptr, c1tensor);
	ASSERT_NE(nullptr, c2tensor);

	std::vector<double> n1out = expose<double>(n1);
	std::vector<double> c1out = c1tensor->expose();
	std::vector<double> c2out = c2tensor->expose();
	EXPECT_TRUE(std::equal(n1out.begin(), n1out.end(), c1out.begin()));
	EXPECT_TRUE(std::equal(expectc2, expectc2 + nc2s, c2out.begin()));
	// bad status returns undefined shapes
	EXPECT_EQ(nullptr, conn3->eval()); // not part defined is undefined
	EXPECT_EQ(nullptr, conn4->eval());
	EXPECT_EQ(nullptr, conn5->eval());

	delete[] expectc2;
	delete conn;
	delete conn2;
	delete conn3;
	delete conn4;
	delete conn5;
	delete n1;
	delete n2;
	delete n3;
	delete n4;
}


TEST_F(IMMUTABLE, ImmutableDeath_I005)
{
	size_t nnodes = get_int(1, "nnodes", nnodes_range)[0];
	std::unordered_set<elem_op*> leaves;
	std::unordered_set<elem_op*> collector;

	// build a tree out of mock immutables
	build_ntree<elem_op >(2, nnodes,
	[this, &leaves](void)
	{
		std::string llabel = get_string(get_int(1, "llabel.size", {14, 29})[0], "llabel");
		elem_op* im = new mock_elem_op(std::vector<inode*>{}, llabel, get_testshaper(this));
		leaves.emplace(im);
		return im;
	},
	[this, &collector](std::vector<elem_op*> args)
	{
		std::string nlabel = get_string(get_int(1, "nlabel.size", {14, 29})[0], "nlabel");
		mock_elem_op* im = new mock_elem_op(
			std::vector<inode*>(args.begin(), args.end()), nlabel, get_testshaper(this));
		im->triggerOnDeath =
		[&collector](mock_elem_op* ded)
		{
			collector.erase(ded);
		};
		collector.insert(im);
		return im;
	});

	// check if collectors are all dead
	for (elem_op* l : leaves)
	{
		delete l;
	}

	EXPECT_TRUE(collector.empty());
	for (elem_op* im : collector)
	{
		delete im;
	}
}


TEST_F(IMMUTABLE, TemporaryEval_I006)
{
	size_t nnodes = get_int(1, "nnodes", nnodes_range)[0];

	std::unordered_set<inode*> leaves;
	std::unordered_set<elem_op*> collector;

	tensorshape shape = random_def_shape(this);
	double single_rando = get_double(1, "single_rando", {1.1, 2.2})[0];

	auto unifiedshaper =
	[&shape](std::vector<tensorshape>)
	{
		return shape;
	};

	const_init cinit(single_rando);

	inode* root = build_ntree<inode >(2, nnodes,
	[this, &leaves, &shape, &cinit]() -> inode*
	{
		std::string llabel = get_string(get_int(1, "llabel.size", {14, 29})[0], "llabel");
		variable* im = new variable(shape, cinit, nnet::DOUBLE, llabel);
		im->initialize();
		leaves.emplace(im);
		return im;
	},
	[this, &collector, &unifiedshaper](std::vector<inode*> args)
	{
		std::string nlabel = get_string(get_int(1, "nlabel.size", {14, 29})[0], "nlabel");
		mock_elem_op* im = new mock_elem_op(args, nlabel, unifiedshaper, adder);
		im->triggerOnDeath =
		[&collector](mock_elem_op* ded)
		{
			collector.erase(ded);
		};
		collector.insert(im);
		return im;
	});

	inode* out = nullptr;
	std::unordered_set<ileaf*> lcache;
	for (elem_op* coll : collector)
	{
		if (coll == root) continue;
		lcache.clear();
		static_cast<elem_op*>(root)->temporary_eval(coll, out);
		ASSERT_NE(nullptr, out);
		const tensor_double* outt = dynamic_cast<const tensor_double*>(out->eval());
		ASSERT_NE(nullptr, outt);
		ASSERT_TRUE(tensorshape_equal(shape, outt->get_shape()));
		// out data should be 1 + M * single_rando where M is the
		// number of root's leaves that are not in coll's leaves
		lcache = coll->get_leaves();
		size_t M = leaves.size() - lcache.size();
		double datum = M * single_rando + 1;
		std::vector<double> odata = outt->expose();
		double diff = std::abs(datum - odata[0]);
		EXPECT_GT(0.000001 * single_rando, diff); // allow error of a tiny fraction of the random leaf value
		delete out;
		out = nullptr;
	}

	for (inode* l : leaves)
	{
		delete l;
	}
	for (elem_op* im : collector)
	{
		delete im;
	}
}


TEST_F(IMMUTABLE, GetLeaves_I007)
{
	size_t nnodes = get_int(1, "nnodes", nnodes_range)[0];

	std::unordered_set<variable*> leaves;
	std::unordered_set<elem_op*> collector;

	inode* root = build_ntree<inode >(2, nnodes,
		[this, &leaves]() -> inode*
		{
			std::string llabel = get_string(get_int(1, "llabel.size", {14, 29})[0], "llabel");
			double leafvalue = get_double(1, "leafvalue")[0];
			variable* im = new variable(leafvalue, llabel);
			leaves.emplace(im);
			return im;
		},
		[this, &collector](std::vector<inode*> args) -> inode*
		{
			std::string nlabel = get_string(get_int(1, "nlabel.size", {14, 29})[0], "nlabel");
			mock_elem_op* im = new mock_elem_op(
				std::vector<inode*>(args.begin(), args.end()), nlabel, get_testshaper(this));
			im->triggerOnDeath =
				[&collector](mock_elem_op* ded) {
					collector.erase(ded);
				};
			collector.insert(im);
			return im;
		});

	// the root has all leaves
	std::unordered_set<ileaf*> lcache = root->get_leaves();
	for (variable* l : leaves)
	{
		EXPECT_TRUE(lcache.end() != lcache.find(l));
	}
	// any collector's leaf is found in leaves (ensures lcache doesn't collect trash nodes)
	for (elem_op* coll : collector)
	{
		lcache.clear();
		lcache = coll->get_leaves();
		for (ileaf* useful : lcache)
		{
			if (variable* uvar = dynamic_cast<variable*>(useful))
			{
				EXPECT_TRUE(leaves.end() != leaves.find(uvar));
			}
		}
	}

	for (variable* l : leaves)
	{
		delete l;
	}
	for (elem_op* im : collector)
	{
		delete im;
	}
}


TEST_F(IMMUTABLE, GetLeaf_I008)
{
	std::vector<iconnector*> ordering;
	mock_node exposer;

	BACK_MAP backer =
	[&ordering](std::vector<std::pair<inode*,inode*>> args) -> inode*
	{
		inode* leef = args[0].second;
		double lvalue = expose<double>(leef)[0];
		if (lvalue == 0.0 && args.size()> 1)
		{
			leef = args[1].second;
			if (iconnector* conn = dynamic_cast<iconnector*>(args[1].first))
			{
				ordering.push_back(conn);
			}
		}
		else if (iconnector* conn = dynamic_cast<iconnector*>(args[0].first))
		{
			ordering.push_back(conn);
		}
		return leef;
	};

	size_t nnodes = get_int(1, "nnodes", nnodes_range)[0];

	std::unordered_set<variable*> leaves;
	std::unordered_set<elem_op*> collector;

	inode* root = build_ntree<inode >(2, nnodes,
		[this, &leaves]() -> inode*
		{
			std::string llabel = get_string(get_int(1, "llabel.size", {14, 29})[0], "llabel");
			double leafvalue = get_double(1, "leafvalue")[0];
			variable* im = new variable(leafvalue, llabel);
			leaves.emplace(im);
			return im;
		},
		[this, &collector, &backer](std::vector<inode*> args) -> inode*
		{
			std::string nlabel = get_string(get_int(1, "nlabel.size", {14, 29})[0], "nlabel");
			mock_elem_op* im = new mock_elem_op(args, nlabel,
				get_testshaper(this), test_abuilder, backer);
			im->triggerOnDeath =
				[&collector](mock_elem_op* ded) {
					collector.erase(ded);
				};
			collector.insert(im);
			return im;
		});

	variable* notleaf = new variable(0);
	for (size_t i = 0; i < nnodes/3; i++)
	{
		ordering.clear();
		variable* l = *(rand_select<std::unordered_set<variable*>>(leaves));
		varptr wun = exposer.expose_leaf(root, l);
		EXPECT_TRUE(bottom_up(ordering));
		ordering.clear();
		varptr zaro = exposer.expose_leaf(root, notleaf);
		EXPECT_TRUE(bottom_up(ordering));

		double value1 = expose<double>(wun)[0];
		double value0 = expose<double>(zaro)[0];
		EXPECT_TRUE(value1 == 1.0);
		EXPECT_TRUE(value0 == 0.0);
	}

	delete notleaf;
	for (variable* l : leaves)
	{
		delete l;
	}
	for (elem_op* im : collector)
	{
		delete im;
	}
}


TEST_F(IMMUTABLE, GetGradient_I009)
{
	tensorshape shape = random_def_shape(this);
	double single_rando = get_double(1, "single_rando", {1.1, 2.2})[0];

	auto unifiedshaper =
	[&shape](std::vector<tensorshape>)
	{
		return shape;
	};

	const_init cinit(single_rando);

	std::vector<iconnector*> ordering;
	BACK_MAP backer =
	[&ordering](std::vector<std::pair<inode*,inode*>> args) -> inode*
	{
		varptr leef = args[0].second;
		double d = expose<double>(leef.get())[0];
		if (d == 0.0 && args.size()> 1)
		{
			leef = args[1].second;
			if (iconnector* conn = dynamic_cast<iconnector*>(args[1].first))
			{
				ordering.push_back(conn);
			}
		}
		else if (iconnector* conn = dynamic_cast<iconnector*>(args[0].first))
		{
			ordering.push_back(conn);
		}
		return leef;
	};

	size_t nnodes = get_int(1, "nnodes", nnodes_range)[0];

	std::unordered_set<variable*> leaves;
	std::unordered_set<elem_op*> collector;

	inode* root = build_ntree<inode >(2, nnodes,
	[this, &leaves, &shape, &cinit]() -> inode*
	{
		std::string llabel = get_string(get_int(1, "llabel.size", {14, 29})[0], "llabel");
		variable* im = new variable(shape, cinit, nnet::DOUBLE, llabel);
		im->initialize();
		leaves.emplace(im);
		return im;
	},
	[this, &collector, &unifiedshaper, &backer](std::vector<inode*> args) -> inode*
	{
		std::string nlabel = get_string(get_int(1, "nlabel.size", {14, 29})[0], "nlabel");
		mock_elem_op* im = new mock_elem_op(args, nlabel,
			unifiedshaper, adder, backer);
		im->triggerOnDeath =
			[&collector](mock_elem_op* ded) {
				collector.erase(ded);
			};
		collector.insert(im);
		return im;
	});

	variable* notleaf = new variable(0);
	std::unordered_set<ileaf*> lcache;
	for (size_t i = 0; i < nnodes/3; i++)
	{
		ordering.clear();
		variable* rselected = *(rand_select<std::unordered_set<variable*>>(leaves));
		const tensor_double* wun =
			dynamic_cast<const tensor_double*>(root->derive(rselected)->eval());
		EXPECT_TRUE(bottom_up(ordering));
		ordering.clear();
		const tensor_double* zaro =
			dynamic_cast<const tensor_double*>(root->derive(notleaf)->eval());
		EXPECT_TRUE(bottom_up(ordering));
		ordering.clear();

		ASSERT_NE(nullptr, wun);
		ASSERT_NE(nullptr, zaro);
		EXPECT_EQ(1, wun->expose()[0]);
		EXPECT_EQ(0, zaro->expose()[0]);

		// SAME AS TEMPORARY EVAL
		elem_op* coll = *(rand_select<std::unordered_set<elem_op*>>(collector));
		if (coll == root) continue;
		const tensor_double* grad_too = dynamic_cast<const tensor_double*>(root->derive(coll)->eval());
		EXPECT_TRUE(bottom_up(ordering));
		ASSERT_NE(nullptr, grad_too);
		ASSERT_TRUE(tensorshape_equal(shape, grad_too->get_shape()));
		// out data should be 1 + M * single_rando where M is the
		// number of root's leaves that are not in coll's leaves
		lcache.clear();
		lcache = coll->get_leaves();
		size_t M = leaves.size() - lcache.size();
		double datum = M * single_rando + 1;
		std::vector<double> odata = grad_too->expose();
		double diff = std::abs(datum - odata[0]);
		EXPECT_GT(0.000001 * single_rando, diff); // allow error of a tiny fraction of the random leaf value
	}

	delete notleaf;
	for (variable* l : leaves)
	{
		delete l;
	}
	for (elem_op* im : collector)
	{
		delete im;
	}
}


TEST_F(IMMUTABLE, Update_I010)
{
	std::string conname = get_string(get_int(1, "conname.size", {14, 29})[0], "conname");
	std::string label1 = get_string(get_int(1, "label1.size", {14, 29})[0], "label1");

	mock_node* n1 = new mock_node(label1);
	tensorshape n1s = random_def_shape(this);
	n1->data_ = new mock_itensor(this, n1s);

	// for this test, we care about data, grab the largest shape, and sum all data that fit in said array
	auto grabs = [](std::vector<tensorshape> ts)
	{
		return ts[0];
	};

	bool mutate = false;
	CONN_ACTOR asis = [&mutate](out_wrapper<void>& dest,
		std::vector<in_wrapper<void> >& srcs,
		nnet::TENS_TYPE type) -> itens_actor*
	{
		return new cond_actor(dest, srcs, mutate);
	};

	elem_op* conn = new mock_elem_op({n1}, conname, grabs, asis);
	std::vector<double> init = expose<double>(conn);
	mutate = true;
	conn->update();
	std::vector<double> next = expose<double>(conn);
	ASSERT_EQ(init.size(), next.size());
	for (size_t i = 0, n = init.size(); i < n; i++)
	{
		EXPECT_EQ(init[i]+1, next[i]);
	}

	delete conn;
	delete n1;
}


TEST_F(IMMUTABLE, ShapeIncompatible_I011)
{
	std::string conname = get_string(get_int(1, "conname.size", {14, 29})[0], "conname");
	std::string conname2 = get_string(get_int(1, "conname2.size", {14, 29})[0], "conname2");
	std::string label1 = get_string(get_int(1, "label1.size", {14, 29})[0], "label1");

	mock_node* n1 = new mock_node(label1);
	tensorshape n1s = random_def_shape(this);
	std::vector<size_t> temp = n1s.as_list();
	temp.push_back(3);
	tensorshape n2s = temp;
	n1->data_ = new mock_itensor(this, n1s);

	bool change = false;
	auto shiftyshaper =
	[&change, n2s](std::vector<tensorshape> ts)
	{
		if (change) return n2s;
		return ts[0];
	};

	mock_elem_op* initialgood = new mock_elem_op({n1}, conname2, shiftyshaper);
	change = true;
	itens_actor*& actor = initialgood->get_actor();
	delete actor;
	actor = nullptr;
	EXPECT_THROW(n1->notify(nnet::notification::UPDATE), std::exception);

	delete initialgood;
	delete n1;
}


#endif /* DISABLE_IMMUTABLE_TEST */


#endif /* DISABLE_CONNECTOR_MODULE_TESTS */
