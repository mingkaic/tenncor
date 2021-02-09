
#ifndef DISABLE_TEQ_TRAVELER_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "internal/teq/mock/mock.hpp"


using ::testing::Const;
using ::testing::Return;


TEST(TRAVELER, GraphStat)
{
	auto a = make_var(teq::Shape());
	auto b = make_var(teq::Shape());
	auto c = make_var(teq::Shape());

	auto d = make_fnc("MOCK2", 0, teq::TensptrsT{c});
	auto f = make_fnc("MOCK1", 0, teq::TensptrsT{a, b});
	auto g = make_fnc("MOCK0", 0, teq::TensptrsT{d, f});

	teq::GraphStat stat;
	g->accept(stat);
	EXPECT_EQ(2, stat.graphsize_[g.get()].upper_);
	EXPECT_EQ(1, stat.graphsize_[f.get()].upper_);
	EXPECT_EQ(0, stat.graphsize_[c.get()].upper_);
	EXPECT_EQ(0, stat.graphsize_[a.get()].upper_);
	EXPECT_EQ(0, stat.graphsize_[b.get()].upper_);
}


TEST(TRAVELER, PathFinder)
{
	auto a = make_var(teq::Shape());
	auto b = make_var(teq::Shape());
	auto c = make_var(teq::Shape());

	auto d = make_fnc("MOCK2", 0, teq::TensptrsT{c});
	auto f = make_fnc("MOCK1", 0, teq::TensptrsT{a, b});
	auto g = make_fnc("MOCK0", 0, teq::TensptrsT{d, f});
	EXPECT_CALL(*d, ls_attrs()).WillRepeatedly(Return(types::StringsT{}));
	EXPECT_CALL(*f, ls_attrs()).WillRepeatedly(Return(types::StringsT{}));
	EXPECT_CALL(*g, ls_attrs()).WillRepeatedly(Return(types::StringsT{}));
	EXPECT_CALL(*d, size()).WillRepeatedly(Return(0));
	EXPECT_CALL(*f, size()).WillRepeatedly(Return(0));
	EXPECT_CALL(*g, size()).WillRepeatedly(Return(0));

	std::string target_key = "target";

	teq::PathFinder finder(teq::TensMapT<std::string>{
		{a.get(), target_key},
		{d.get(), target_key + "2"}
	});
	g->accept(finder);

	{
		ASSERT_HAS(finder.roadmap_, g.get());
		auto& gdirs = finder.roadmap_[g.get()];
		ASSERT_HAS(gdirs, "target");
		EXPECT_EQ(1, gdirs["target"].args_.size());
		EXPECT_ARRHAS(gdirs["target"].args_, 1);

		ASSERT_HAS(gdirs, "target2");
		EXPECT_EQ(1, gdirs["target2"].args_.size());
		EXPECT_ARRHAS(gdirs["target2"].args_, 0);

		ASSERT_HAS(finder.roadmap_, f.get());
		auto& fdirs = finder.roadmap_[f.get()];
		ASSERT_HAS(fdirs, "target")
		EXPECT_ARRHAS(fdirs["target"].args_, 0);

		ASSERT_HASNOT(finder.roadmap_, d.get());
	}

	finder.clear();
	f->accept(finder);

	{
		ASSERT_HASNOT(finder.roadmap_, g.get());

		ASSERT_HAS(finder.roadmap_, f.get());
		auto& fdirs = finder.roadmap_[f.get()];
		ASSERT_HAS(fdirs, "target")
		EXPECT_ARRHAS(fdirs["target"].args_, 0);
	}

	teq::PathFinder finder2(
		teq::TensMapT<std::string>{{c.get(), target_key}});
	g->accept(finder2);

	{
		ASSERT_HAS(finder2.roadmap_, g.get());
		auto& gdirs = finder2.roadmap_[g.get()];
		ASSERT_HAS(gdirs, "target")
		EXPECT_ARRHAS(gdirs["target"].args_, 0);
	}

	finder2.clear();
	f->accept(finder2);

	EXPECT_HASNOT(finder2.roadmap_, f.get());
	EXPECT_EQ(0, finder2.roadmap_.size());
}


TEST(TRAVELER, PathFinderAttr)
{
	auto a = make_var(teq::Shape());
	auto b = make_var(teq::Shape());
	auto c = make_var(teq::Shape());

	auto d = make_fnc("MOCK2", 0, teq::TensptrsT{b});
	auto f = make_fnc("MOCK1", 0, teq::TensptrsT{a, c});
	auto g = make_fnc("MOCK0", 0, teq::TensptrsT{a, d});
	EXPECT_CALL(*d, ls_attrs()).WillRepeatedly(Return(types::StringsT{"yodoo"}));
	EXPECT_CALL(*f, ls_attrs()).WillRepeatedly(Return(types::StringsT{}));
	EXPECT_CALL(*g, ls_attrs()).WillRepeatedly(Return(types::StringsT{"numbers", "tensors"}));
	EXPECT_CALL(*d, size()).WillRepeatedly(Return(1));
	EXPECT_CALL(*f, size()).WillRepeatedly(Return(0));
	EXPECT_CALL(*g, size()).WillRepeatedly(Return(2));

	teq::TensorObj ctens(c);
	marsh::Number<double> numb(333.4);
	teq::TensorObj ftens(f);

	EXPECT_CALL(*d, get_attr("yodoo")).WillRepeatedly(Return(&ctens));
	EXPECT_CALL(*g, get_attr("numbers")).WillRepeatedly(Return(&numb));
	EXPECT_CALL(*g, get_attr("tensors")).WillRepeatedly(Return(&ftens));
	EXPECT_CALL(Const(*d), get_attr("yodoo")).WillRepeatedly(Return(&ctens));
	EXPECT_CALL(Const(*g), get_attr("numbers")).WillRepeatedly(Return(&numb));
	EXPECT_CALL(Const(*g), get_attr("tensors")).WillRepeatedly(Return(&ftens));

	std::string target_key = "target";

	teq::PathFinder finder(
		teq::TensMapT<std::string>{{c.get(), target_key}});
	g->accept(finder);

	ASSERT_HAS(finder.roadmap_, g.get());
	auto& gdirs = finder.roadmap_[g.get()];
	ASSERT_HAS(gdirs, target_key);
	EXPECT_EQ(1, gdirs[target_key].attrs_.size());
	ASSERT_ARRHAS(gdirs[target_key].attrs_, "tensors");

	ASSERT_HAS(finder.roadmap_, d.get());
	auto& ddirs = finder.roadmap_[d.get()];
	ASSERT_HAS(ddirs, target_key);
	EXPECT_TRUE(ddirs[target_key].args_.empty());
	EXPECT_EQ(1, ddirs[target_key].attrs_.size());
	ASSERT_ARRHAS(ddirs[target_key].attrs_, "yodoo");
}


TEST(TRAVELER, ParentGraph)
{
	auto a = make_var(teq::Shape());
	auto b = make_var(teq::Shape());
	auto c = make_var(teq::Shape());

	auto f = make_fnc("MOCK2", 0, teq::TensptrsT{a, b});
	auto g = make_fnc("MOCK1", 0, teq::TensptrsT{f, b});
	auto h = make_fnc("MOCK0", 0, teq::TensptrsT{c, f, g});
	EXPECT_CALL(*f, ls_attrs()).WillRepeatedly(Return(types::StringsT{}));
	EXPECT_CALL(*g, ls_attrs()).WillRepeatedly(Return(types::StringsT{}));
	EXPECT_CALL(*h, ls_attrs()).WillRepeatedly(Return(types::StringsT{}));
	EXPECT_CALL(*f, size()).WillRepeatedly(Return(0));
	EXPECT_CALL(*g, size()).WillRepeatedly(Return(0));
	EXPECT_CALL(*h, size()).WillRepeatedly(Return(0));

	teq::ParentFinder finder;
	h->accept(finder);

	// expect: a -> [f], b -> [f, g], c -> [h], f -> [g, h], g -> [h], h -> []
	auto& parents = finder.parents_;
	auto aparents = parents[a.get()];
	auto bparents = parents[b.get()];
	auto cparents = parents[c.get()];
	auto fparents = parents[f.get()];
	auto gparents = parents[g.get()];
	auto hparents = parents[h.get()];

	EXPECT_EQ(1, aparents.size());
	EXPECT_EQ(2, bparents.size());
	EXPECT_EQ(1, cparents.size());
	EXPECT_EQ(2, fparents.size());
	EXPECT_EQ(1, gparents.size());
	EXPECT_EQ(0, hparents.size());

	EXPECT_HAS(aparents, f.get());
	EXPECT_HAS(bparents, f.get());
	EXPECT_HAS(bparents, g.get());
	EXPECT_HAS(cparents, h.get());
	EXPECT_HAS(fparents, g.get());
	EXPECT_HAS(fparents, h.get());
	EXPECT_HAS(gparents, h.get());
}


TEST(TRAVELER, ParentFinderAttr)
{
	auto a = make_var(teq::Shape());
	auto b = make_var(teq::Shape());
	auto c = make_var(teq::Shape());

	auto d = make_fnc("MOCK2", 0, teq::TensptrsT{b});
	auto f = make_fnc("MOCK1", 0, teq::TensptrsT{a, c});
	auto g = make_fnc("MOCK0", 0, teq::TensptrsT{a, d});
	EXPECT_CALL(*d, ls_attrs()).WillRepeatedly(Return(types::StringsT{"yodoo"}));
	EXPECT_CALL(*f, ls_attrs()).WillRepeatedly(Return(types::StringsT{}));
	EXPECT_CALL(*g, ls_attrs()).WillRepeatedly(Return(types::StringsT{"numbers", "tensors"}));
	EXPECT_CALL(*d, size()).WillRepeatedly(Return(1));
	EXPECT_CALL(*f, size()).WillRepeatedly(Return(0));
	EXPECT_CALL(*g, size()).WillRepeatedly(Return(2));

	teq::TensorObj ctens(c);
	marsh::Number<double> numb(333.4);
	teq::TensorObj ftens(f);

	EXPECT_CALL(*d, get_attr("yodoo")).WillRepeatedly(Return(&ctens));
	EXPECT_CALL(*g, get_attr("numbers")).WillRepeatedly(Return(&numb));
	EXPECT_CALL(*g, get_attr("tensors")).WillRepeatedly(Return(&ftens));
	EXPECT_CALL(Const(*d), get_attr("yodoo")).WillRepeatedly(Return(&ctens));
	EXPECT_CALL(Const(*g), get_attr("numbers")).WillRepeatedly(Return(&numb));
	EXPECT_CALL(Const(*g), get_attr("tensors")).WillRepeatedly(Return(&ftens));

	std::string target_key = "target";

	teq::ParentFinder finder(
		[](teq::iFunctor& func)
		{
			auto deps = func.get_args();
			if (func.size() > 0)
			{
				marsh::Maps attrs;
				marsh::get_attrs(attrs, func);

				teq::FindTensAttr finder;
				attrs.accept(finder);
				deps.insert(deps.end(),
					finder.tens_.begin(), finder.tens_.end());
			}
			return deps;
		});
	g->accept(finder);

	auto& parents = finder.parents_;
	auto aparents = parents[a.get()];
	auto bparents = parents[b.get()];
	auto cparents = parents[c.get()];
	auto dparents = parents[d.get()];
	auto fparents = parents[f.get()];
	auto gparents = parents[g.get()];

	EXPECT_EQ(2, aparents.size());
	EXPECT_EQ(1, bparents.size());
	EXPECT_EQ(2, cparents.size());
	EXPECT_EQ(1, dparents.size());
	EXPECT_EQ(1, fparents.size());
	EXPECT_EQ(0, gparents.size());
}


TEST(TRAVELER, Owners)
{
	auto a = make_var(teq::Shape());
	auto b = make_var(teq::Shape());
	auto c = make_var(teq::Shape());

	teq::RefMapT owners;
	teq::iTensor* fref;
	teq::iTensor* gref;
	{
		auto f = make_fnc("f", 0, teq::TensptrsT{a, b});
		auto g = make_fnc("g", 0, teq::TensptrsT{f, c});
		EXPECT_CALL(*f, ls_attrs()).WillRepeatedly(Return(types::StringsT{}));
		EXPECT_CALL(*g, ls_attrs()).WillRepeatedly(Return(types::StringsT{}));
		EXPECT_CALL(*f, size()).WillRepeatedly(Return(0));
		EXPECT_CALL(*g, size()).WillRepeatedly(Return(0));

		fref = f.get();
		gref = g.get();

		owners = teq::track_ownrefs(teq::TensptrsT{g});
		ASSERT_HAS(owners, a.get());
		ASSERT_HAS(owners, b.get());
		ASSERT_HAS(owners, c.get());
		ASSERT_HAS(owners, fref);
		ASSERT_HAS(owners, gref);

		EXPECT_FALSE(owners[a.get()].expired());
		EXPECT_FALSE(owners[b.get()].expired());
		EXPECT_FALSE(owners[c.get()].expired());
		EXPECT_FALSE(owners[fref].expired());
		EXPECT_FALSE(owners[gref].expired());

		auto alocked = owners[a.get()].lock();
		auto blocked = owners[b.get()].lock();
		auto clocked = owners[c.get()].lock();
		auto flocked = owners[fref].lock();
		auto glocked = owners[gref].lock();
		EXPECT_EQ(a.use_count(), alocked.use_count());
		EXPECT_EQ(b.use_count(), blocked.use_count());
		EXPECT_EQ(c.use_count(), clocked.use_count());
		EXPECT_EQ(f.use_count(), flocked.use_count());
		EXPECT_EQ(g.use_count(), glocked.use_count());
	}

	EXPECT_FALSE(owners[a.get()].expired());
	EXPECT_FALSE(owners[b.get()].expired());
	EXPECT_FALSE(owners[c.get()].expired());
	EXPECT_TRUE(owners[fref].expired());
	EXPECT_TRUE(owners[gref].expired());

	auto master = teq::convert_ownmap(owners);
	ASSERT_EQ(3, master.size());
	EXPECT_HAS(master, a.get());
	EXPECT_HAS(master, b.get());
	EXPECT_HAS(master, c.get());
}


#endif // DISABLE_TEQ_TRAVELER_TEST
