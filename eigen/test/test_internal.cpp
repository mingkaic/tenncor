
#ifndef DISABLE_INTERNAL_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "teq/mock/leaf.hpp"

#include "eigen/operator.hpp"

#include "eigen/mock/edge.hpp"


TEST(INTERNAL, PairEncodeDecode)
{
	eigen::PairVecT<double> pairs = {{1, 2}, {3, 4}, {5, 6}};

	auto vecs = eigen::encode_pair(pairs);
	auto apairs = eigen::decode_pair<size_t>(vecs);
	ASSERT_EQ(pairs.size(), apairs.size());
	for (size_t i = 0; i < 3; ++i)
	{
		auto orig = pairs[i];
		auto apair = apairs[i];
		EXPECT_EQ(orig.first, apair.first);
		EXPECT_EQ(orig.second, apair.second);
	}

	std::vector<double> bad = {1, 2, 3, 4, 5};
	EXPECT_FATAL(eigen::decode_pair<size_t>(bad),
		"cannot decode odd vector [1\\2\\3\\4\\5] into vec of pairs");
}


TEST(INTERNAL, Conversions)
{
	std::vector<double> values = {
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


TEST(INTERNAL, MakeEigenmap)
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


template <typename T>
struct JunkEdge final : public eigen::iEigenEdge<T>
{
	JunkEdge (void) = default;

	/// Implementation of iEdge
	teq::Shape shape (void) const override
	{
		return teq::Shape();
	}

	/// Implementation of iEdge
	teq::Shape argshape (void) const override
	{
		return teq::Shape();
	}

	/// Implementation of iEdge
	teq::TensptrT get_tensor (void) const override
	{
		return nullptr;
	}

	/// Implementation of iEdge
	void get_attrs (marsh::Maps& out) const override
	{
		auto arr = std::make_unique<marsh::NumArray<size_t>>();
		arr->contents_ = {1, 2, 3};
		out.contents_.emplace("coorder", std::move(arr));
	}

	T* data (void) const override
	{
		return nullptr;
	}
};


TEST(INTERNAL, GetCoorder)
{
	std::vector<double> coord{3, 4, 55};
	teq::Shape shape({2, 3});

	MockEdge<double> edge(
		teq::TensptrT(new MockTensor(shape)),
		std::vector<double>{2, 8, 4, 5, 7, 6}, shape, coord);
	auto c = get_coorder(edge);
	EXPECT_VECEQ(coord, c);

	MockEdge<double> badedge(
		teq::TensptrT(new MockTensor(teq::Shape(shape))),
		std::vector<double>{2, 8, 4, 5, 7, 6}, shape);
	EXPECT_FATAL(get_coorder(badedge), "coorder not found");

	JunkEdge<double> worstedge;
	EXPECT_FATAL(get_coorder(worstedge), "cannot find array coorder");
}


#endif // DISABLE_INTERNAL_TEST
