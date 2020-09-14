
#ifndef DISABLE_EIGEN_PACKER_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "internal/teq/mock/mock.hpp"

#include "internal/eigen/operator.hpp"


TEST(PACKER, PairEncodeDecode)
{
	eigen::PairVecT<double> pairs = {{1.2, 2.3}, {3.4, 4.6}, {5.5, 6.7}};

	auto vecs = eigen::encode_pair(pairs);
	auto apairs = eigen::decode_pair<double>(vecs);
	size_t n = pairs.size();
	ASSERT_EQ(n, apairs.size());
	for (size_t i = 0; i < n; ++i)
	{
		auto orig = pairs[i];
		auto apair = apairs[i];
		EXPECT_EQ(int64_t(orig.first), apair.first);
		EXPECT_EQ(int64_t(orig.second), apair.second);
	}

	std::vector<int64_t> bad = {1, 2, 3, 4, 5};
	EXPECT_FATAL(eigen::decode_pair<size_t>(bad),
		"cannot decode odd vector [1\\2\\3\\4\\5] into vec of pairs");

	EXPECT_STREQ("[1:2\\3:4\\5:6]", eigen::to_string(pairs).c_str());
}


TEST(PACKER, Conversions)
{
	std::vector<int64_t> values = {
		1, 2, 4, 8, 16, 32, 64, 128, 256};
	eigen::TensorT<double> a(3, 3, 1, 1, 1, 1, 1, 1);
	std::copy(values.begin(), values.end(), a.data());

	auto b = eigen::tens_to_matmap(a);

	auto c = eigen::tens_to_tensmap(a);

	auto d = eigen::tensmap_to_matmap(c);

	auto rawb = b.data();
	auto rawc = c.data();
	auto rawd = d.data();

	std::vector<double> gotb(rawb, rawb + 9);
	std::vector<double> gotc(rawc, rawc + 9);
	std::vector<double> gotd(rawd, rawd + 9);

	EXPECT_VECEQ(values, gotb);
	EXPECT_VECEQ(values, gotc);
	EXPECT_VECEQ(values, gotd);

	eigen::MatrixT<double> m(3, 3);
	std::copy(values.begin(), values.end(), m.data());

	auto e = eigen::mat_to_matmap(m);

	auto f = eigen::mat_to_tensmap(m);

	auto rawe = e.data();
	auto rawf = f.data();

	std::vector<double> gote(rawe, rawe + 9);
	std::vector<double> gotf(rawf, rawf + 9);

	EXPECT_VECEQ(values, gote);
	EXPECT_VECEQ(values, gotf);
}


TEST(PACKER, MakeEigenmap)
{
	std::vector<double> a = {1, 2, 3, 4, 5, 6};
	teq::Shape shape({2, 3});
	eigen::MatMapT<double> mat = eigen::make_matmap<double>(a.data(), shape);
	eigen::TensMapT<double> tmap = eigen::make_tensmap<double>(a.data(), shape);
	eigen::TensorT<double> tens = eigen::make_tensmap<double>(a.data(), shape);

	double* data = mat.data();
	double* mdata = tmap.data();
	double* tdata = tens.data();

	a[2] = 9;

	std::vector<double> expect = {1, 2, 9, 4, 5, 6};
	std::vector<double> got_mat(data, data + shape.n_elems());
	std::vector<double> got_mten(mdata, mdata + shape.n_elems());
	std::vector<double> got_tens(tdata, tdata + shape.n_elems());
	EXPECT_VECEQ(expect, got_mat);
	EXPECT_VECEQ(expect, got_mten);

	std::vector<double> expect_static = {1, 2, 3, 4, 5, 6};
	EXPECT_VECEQ(expect_static, got_tens);

	teq::Shape mshape = eigen::get_shape<double>(tmap);
	teq::Shape tshape = eigen::get_shape<double>(tens);
	EXPECT_ARREQ(shape, mshape);
	EXPECT_ARREQ(shape, tshape);

	EXPECT_FATAL(eigen::make_matmap<double>(nullptr, teq::Shape()),
		"cannot get matmap from nullptr");
	EXPECT_FATAL(eigen::make_tensmap<double>(nullptr, teq::Shape()),
		"cannot get tensmap from nullptr");
}


TEST(PACKER, BadPacker)
{
	eigen::Packer<std::string> badpack;
	EXPECT_STREQ("", badpack.get_key().c_str());
	marsh::Maps attr;
	EXPECT_FATAL(eigen::get_attr(badpack, attr), "cannot find `` attribute");

	std::string str;
	EXPECT_FATAL(badpack.pack(attr, str), "unknown attribute");
	EXPECT_FATAL(badpack.unpack(str, attr), "unknown attribute");

	global::set_logger(new tutil::NoSupportLogger());
	std::string special_input = "abc///input22z";
	badpack.pack(attr, special_input);
	badpack.unpack(str, attr);
	EXPECT_STRNE(special_input.c_str(), str.c_str());
	EXPECT_STREQ("", str.c_str());
	global::set_logger(new exam::TestLogger());
}


TEST(PACKER, PackerDimPairs)
{
	eigen::PairVecT<teq::DimT> dims = {
		{2, 2},
		{3, 4},
	};
	eigen::PairVecT<teq::DimT> outdims;

	eigen::Packer<eigen::PairVecT<teq::DimT>> packer;
	marsh::Maps attrs;
	EXPECT_FATAL(packer.unpack(outdims, attrs), "cannot find `dimension_pairs` attribute");
	packer.pack(attrs, dims);

	EXPECT_NE(nullptr, attrs.get_attr(packer.get_key()));
	packer.unpack(outdims, attrs);

	size_t n = dims.size();
	ASSERT_EQ(n, outdims.size());
	for (size_t i = 0; i < n; ++i)
	{
		auto orig = dims[i];
		auto apair = outdims[i];
		EXPECT_EQ(orig.first, apair.first);
		EXPECT_EQ(orig.second, apair.second);
	}

	marsh::Maps attrs2;
	eigen::pack_attr(attrs2, dims);
	EXPECT_NE(nullptr, attrs2.get_attr(packer.get_key()));
}


TEST(PACKER, PackerRankPairs)
{
	eigen::PairVecT<teq::RankT> ranks = {
		{2, 2},
		{3, 4},
	};
	eigen::PairVecT<teq::RankT> badranks = {
		{teq::rank_cap, 3},
		{4, teq::rank_cap + 2},
	};
	eigen::PairVecT<teq::RankT> outranks;

	eigen::Packer<eigen::PairVecT<teq::RankT>> packer;
	marsh::Maps attrs;
	EXPECT_FATAL(packer.unpack(outranks, attrs), "cannot find `rank_pairs` attribute");
	EXPECT_FATAL(packer.pack(attrs, badranks),
		"cannot reference ranks beyond rank_cap 8: [8:3\\4:10]");
	packer.pack(attrs, ranks);

	EXPECT_NE(nullptr, attrs.get_attr(packer.get_key()));
	packer.unpack(outranks, attrs);

	size_t n = ranks.size();
	ASSERT_EQ(n, outranks.size());
	for (size_t i = 0; i < n; ++i)
	{
		auto orig = ranks[i];
		auto apair = outranks[i];
		EXPECT_EQ(orig.first, apair.first);
		EXPECT_EQ(orig.second, apair.second);
	}

	marsh::Maps attrs2;
	eigen::pack_attr(attrs2, ranks);
	EXPECT_NE(nullptr, attrs2.get_attr(packer.get_key()));
}


TEST(PACKER, PackerDims)
{
	teq::DimsT dims = {2, 2, 3, 4};
	teq::DimsT outdims;

	eigen::Packer<teq::DimsT> packer;
	marsh::Maps attrs;
	EXPECT_FATAL(packer.unpack(outdims, attrs), "cannot find `dimensions` attribute");
	packer.pack(attrs, dims);

	EXPECT_NE(nullptr, attrs.get_attr(packer.get_key()));
	packer.unpack(outdims, attrs);

	size_t n = dims.size();
	ASSERT_EQ(n, outdims.size());
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(dims[i], outdims[i]);
	}

	marsh::Maps attrs2;
	eigen::pack_attr(attrs2, dims);
	EXPECT_NE(nullptr, attrs2.get_attr(packer.get_key()));
}


TEST(PACKER, PackerRanks)
{
	teq::RanksT ranks = {2, 2, 3, 4};
	teq::RanksT badranks = {
		teq::rank_cap, 3, 4, teq::rank_cap + 2};
	teq::RanksT outranks;

	eigen::Packer<teq::RanksT> packer;
	marsh::Maps attrs;
	EXPECT_FATAL(packer.unpack(outranks, attrs), "cannot find `ranks` attribute");
	EXPECT_FATAL(packer.pack(attrs, badranks),
		"cannot reference ranks beyond rank_cap 8: [8\\3\\4\\10]");
	packer.pack(attrs, ranks);

	EXPECT_NE(nullptr, attrs.get_attr(packer.get_key()));
	packer.unpack(outranks, attrs);

	size_t n = ranks.size();
	ASSERT_EQ(n, outranks.size());
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(ranks[i], outranks[i]);
	}

	marsh::Maps attrs2;
	eigen::pack_attr(attrs2, ranks);
	EXPECT_NE(nullptr, attrs2.get_attr(packer.get_key()));
}


TEST(PACKER, PackerRankSet)
{
	std::set<teq::RankT> ranks = {2, 6, 3, 4};
	std::set<teq::RankT> badranks = {
		teq::rank_cap, 3, 4, teq::rank_cap + 2};
	std::set<teq::RankT> bigranks = {1, 2, 3, 4, 5,
		teq::rank_cap, 6, 7, teq::rank_cap + 2};
	std::set<teq::RankT> outranks;

	eigen::Packer<std::set<teq::RankT>> packer;
	marsh::Maps attrs;
	EXPECT_FATAL(packer.unpack(outranks, attrs), "cannot find `rank_set` attribute");
	EXPECT_FATAL(packer.pack(attrs, badranks),
		"cannot reference ranks beyond rank_cap 8: [3\\4\\8\\10]");
	EXPECT_FATAL(packer.pack(attrs, bigranks),
		"cannot specify 9 ranks when 8 (rank_cap) are available");
	packer.pack(attrs, ranks);

	EXPECT_NE(nullptr, attrs.get_attr(packer.get_key()));
	packer.unpack(outranks, attrs);

	EXPECT_VECEQ(ranks, outranks);

	marsh::Maps attrs2;
	eigen::pack_attr(attrs2, ranks);
	EXPECT_NE(nullptr, attrs2.get_attr(packer.get_key()));
}


TEST(PACKER, PackerRank)
{
	teq::RankT rank = 2;
	teq::RankT outrank;

	eigen::Packer<teq::RankT> packer;
	marsh::Maps attrs;
	EXPECT_FATAL(packer.unpack(outrank, attrs), "cannot find `rank` attribute");
	packer.pack(attrs, rank);

	EXPECT_NE(nullptr, attrs.get_attr(packer.get_key()));
	packer.unpack(outrank, attrs);

	EXPECT_EQ(rank, outrank);

	marsh::Maps attrs2;
	eigen::pack_attr(attrs2, rank);
	EXPECT_NE(nullptr, attrs2.get_attr(packer.get_key()));
}


TEST(PACKER, PackerShape)
{
	teq::Shape shape({2, 1, 4});
	teq::Shape outshape;

	eigen::Packer<teq::Shape> packer;
	marsh::Maps attrs;
	EXPECT_FATAL(packer.unpack(outshape, attrs), "cannot find `shape` attribute");
	packer.pack(attrs, shape);

	EXPECT_NE(nullptr, attrs.get_attr(packer.get_key()));
	packer.unpack(outshape, attrs);

	EXPECT_ARREQ(shape, outshape);

	marsh::Maps attrs2;
	eigen::pack_attr(attrs2, shape);
	EXPECT_NE(nullptr, attrs2.get_attr(packer.get_key()));
}


TEST(PACKER, PackerTensor)
{
	teq::TensptrT tens(new MockLeaf());

	eigen::Packer<teq::TensptrT> packer;
	marsh::Maps attrs;
	packer.pack(attrs, tens);

	EXPECT_NE(nullptr, attrs.get_attr(packer.get_key()));
	teq::TensptrT outtens;
	packer.unpack(outtens, attrs);

	EXPECT_EQ(tens, outtens);

	marsh::Maps attrs2;
	eigen::pack_attr(attrs2, tens);
	EXPECT_NE(nullptr, attrs2.get_attr(packer.get_key()));
}


TEST(PACKER, EmptyPacking)
{
	marsh::Maps attrs;
	eigen::pack_attr(attrs);
	EXPECT_EQ(0, attrs.size());
}


TEST(PACKER, ExtendPacking)
{
	teq::Shape inshape({1, 2, 3});
	teq::DimsT extends = {4, 1, 1, 2, 1, 1, 1, 1};
	teq::Shape outshape({4, 2, 3, 2});

	{ // extend by explicit broadcast
		eigen::Packer<teq::DimsT> packer;
		marsh::Maps attrs;
		packer.pack(attrs, extends);

		auto ext = eigen::unpack_extend(inshape, attrs);
		EXPECT_TRUE(ext);
		teq::DimsT xlist = *ext;
		EXPECT_VECEQ(extends, xlist);
	}
	{ // extend by tensor similarity
		teq::TensptrT tens(new MockLeaf(outshape));
		eigen::Packer<teq::TensptrT> packer;
		marsh::Maps attrs;
		packer.pack(attrs, tens);

		auto ext = eigen::unpack_extend(inshape, attrs);
		EXPECT_TRUE(ext);
		teq::DimsT xlist = *ext;
		EXPECT_VECEQ(extends, xlist);
	}
}


#endif // DISABLE_EIGEN_PACKER_TEST
