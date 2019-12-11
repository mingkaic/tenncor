
#ifndef DISABLE_OPT_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "eteq/generated/api.hpp"
#include "eteq/optimize.hpp"


TEST(OPTIMIZE, EqualityCheck)
{
	eteq::Hasher<double> hash;

	teq::Shape shape({2, 3, 4});
	std::vector<double> data = {
		22, 15, 74, 38, 61, 95, 62, 81, 99, 76, 7, 22,
		56, 50, 19, 13, 12, 10, 31, 40, 60, 54, 6, 83
	};
	std::vector<double> data2 = {
		22, 15, 74, 38, 61, 5, 62, 81, 99, 76, 7, 22,
		56, 50, 19, 13, 12, 10, 31, 40, 60, 54, 6, 83
	};

	auto cst = eteq::make_constant<double>(data.data(), shape);
	auto cst2 = eteq::make_constant<double>(data.data(), shape);
	auto notcst = eteq::make_constant<double>(data2.data(), shape);
	auto notcst2 = eteq::make_constant<double>(data.data(), teq::Shape({4, 3, 2}));
	auto var = eteq::to_link<double>(eteq::make_variable<double>(shape));
	auto var2 = eteq::to_link<double>(eteq::make_variable<double>(shape));

	auto p1 = tenncor::permute(notcst2, {2, 1, 0});
	auto p2 = tenncor::permute(notcst2, {1, 2, 0});
	p1->get_tensor()->accept(hash);
	p2->get_tensor()->accept(hash);

	ASSERT_HAS(hash.hashes_, p1->get_tensor().get());
	ASSERT_HAS(hash.hashes_, p2->get_tensor().get());
	auto pid = hash.hashes_[p1->get_tensor().get()];
	EXPECT_NE(pid, hash.hashes_[p2->get_tensor().get()]);

	auto s1 = tenncor::slice(cst, 2, 1, 2);
	auto s2 = tenncor::slice(cst, 0, 1, 2);
	s1->get_tensor()->accept(hash);
	s2->get_tensor()->accept(hash);

	ASSERT_HAS(hash.hashes_, s1->get_tensor().get());
	ASSERT_HAS(hash.hashes_, s2->get_tensor().get());
	EXPECT_NE(
		hash.hashes_[s1->get_tensor().get()],
		hash.hashes_[s2->get_tensor().get()]);

	auto a = cst + cst2;
	auto b = cst2 + cst;
	a->get_tensor()->accept(hash);
	b->get_tensor()->accept(hash);
	ASSERT_HAS(hash.hashes_, a->get_tensor().get());
	ASSERT_HAS(hash.hashes_, b->get_tensor().get());
	EXPECT_EQ(
		hash.hashes_[a->get_tensor().get()],
		hash.hashes_[b->get_tensor().get()]);

	auto c = var / cst;
	auto d = var / cst;
	auto bad = cst / var;
	auto bad2 = var2 / cst;
	c->get_tensor()->accept(hash);
	d->get_tensor()->accept(hash);
	bad->get_tensor()->accept(hash);
	bad2->get_tensor()->accept(hash);
	ASSERT_HAS(hash.hashes_, c->get_tensor().get());
	ASSERT_HAS(hash.hashes_, d->get_tensor().get());
	ASSERT_HAS(hash.hashes_, bad->get_tensor().get());
	ASSERT_HAS(hash.hashes_, bad2->get_tensor().get());
	auto did = hash.hashes_[d->get_tensor().get()];
	EXPECT_EQ(did, hash.hashes_[c->get_tensor().get()]);
	EXPECT_NE(did, hash.hashes_[bad->get_tensor().get()]);
	EXPECT_NE(did, hash.hashes_[bad2->get_tensor().get()]);

	auto e = tenncor::sin(cst);
	auto f = tenncor::sin(cst2);
	auto g = tenncor::sin(notcst);
	e->get_tensor()->accept(hash);
	f->get_tensor()->accept(hash);
	g->get_tensor()->accept(hash);
	ASSERT_HAS(hash.hashes_, e->get_tensor().get());
	ASSERT_HAS(hash.hashes_, f->get_tensor().get());
	ASSERT_HAS(hash.hashes_, g->get_tensor().get());
	auto eid = hash.hashes_[e->get_tensor().get()];
	EXPECT_EQ(eid, hash.hashes_[f->get_tensor().get()]);
	EXPECT_NE(eid, hash.hashes_[g->get_tensor().get()]);

	auto h = cst * cst2;
	auto i = cst2 * cst;
	auto j = notcst * cst;
	h->get_tensor()->accept(hash);
	i->get_tensor()->accept(hash);
	j->get_tensor()->accept(hash);
	ASSERT_HAS(hash.hashes_, h->get_tensor().get());
	ASSERT_HAS(hash.hashes_, i->get_tensor().get());
	ASSERT_HAS(hash.hashes_, j->get_tensor().get());
	auto hid = hash.hashes_[h->get_tensor().get()];
	EXPECT_EQ(hid, hash.hashes_[i->get_tensor().get()]);
	EXPECT_NE(hid, hash.hashes_[j->get_tensor().get()]);

	ASSERT_HAS(hash.hashes_, cst->get_tensor().get());
	ASSERT_HAS(hash.hashes_, cst2->get_tensor().get());
	ASSERT_HAS(hash.hashes_, notcst->get_tensor().get());
	ASSERT_HAS(hash.hashes_, notcst2->get_tensor().get());
	ASSERT_HAS(hash.hashes_, var->get_tensor().get());
	ASSERT_HAS(hash.hashes_, var2->get_tensor().get());
	auto cid = hash.hashes_[cst->get_tensor().get()];
	auto c2id = hash.hashes_[cst2->get_tensor().get()];
	auto notcid = hash.hashes_[notcst->get_tensor().get()];
	auto notc2id = hash.hashes_[notcst2->get_tensor().get()];
	auto vid = hash.hashes_[var->get_tensor().get()];
	auto v2id = hash.hashes_[var2->get_tensor().get()];
	EXPECT_EQ(cid, c2id);
	EXPECT_NE(cid, notcid);
	EXPECT_NE(cid, notc2id);
	EXPECT_NE(vid, cid);
	EXPECT_NE(vid, v2id);
	EXPECT_NE(pid, vid);
}


#endif // DISABLE_OPT_TEST
