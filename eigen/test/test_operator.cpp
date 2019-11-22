
#ifndef DISABLE_OPERATOR_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "teq/mock/leaf.hpp"

#include "eigen/operator.hpp"

#include "eigen/mock/edge.hpp"


static void test_reduce (
	std::function<eigen::EigenptrT<double>(const eigen::iEigenEdge<double>&)> red,
	std::function<double(double,double)> agg)
{
	MockEdge<double> edge(
		teq::TensptrT(new MockTensor(teq::Shape({3, 2}))),
		std::vector<double>{2, 3, 4, 5, 6, 7},
		teq::Shape({3}),
		std::vector<double>{1});
	auto r = red(edge);

	double* raw = r->get_ptr();
	r->assign();
	EXPECT_EQ(agg(2, 5), raw[0]);
	EXPECT_EQ(agg(3, 6), raw[1]);
	EXPECT_EQ(agg(4, 7), raw[2]);
}


TEST(OPERATOR, ReduceSum)
{
	test_reduce(eigen::reduce_sum<double>, [](double a, double b) { return a + b; });
}


TEST(OPERATOR, ReduceProd)
{
	test_reduce(eigen::reduce_prod<double>, [](double a, double b) { return a * b; });
}


TEST(OPERATOR, ReduceMin)
{
	test_reduce(eigen::reduce_min<double>, [](double a, double b) { return std::min(a, b); });
}


TEST(OPERATOR, ReduceMax)
{
	test_reduce(eigen::reduce_max<double>, [](double a, double b) { return std::max(a, b); });
}


TEST(OPERATOR, ArgMax)
{
	MockEdge<double> edge(
		teq::TensptrT(new MockTensor(teq::Shape({3, 2}))),
		std::vector<double>{2, 8, 4, 5, 6, 7},
		teq::Shape({3}),
		std::vector<double>{1});
	auto r = eigen::argmax(edge);

	double* raw = r->get_ptr();
	r->assign();
	EXPECT_EQ(1, raw[0]);
	EXPECT_EQ(0, raw[1]);
	EXPECT_EQ(1, raw[2]);

	MockEdge<double> edge2(
		teq::TensptrT(new MockTensor(teq::Shape({3, 2}))),
		std::vector<double>{2, 8, 4, 5, 9, 7},
		teq::Shape({1}),
		std::vector<double>{8});
	auto r2 = eigen::argmax(edge2);

	double* raw2 = r2->get_ptr();
	r2->assign();
	EXPECT_EQ(4, raw2[0]);
}


TEST(OPERATOR, Extend)
{
	teq::Shape outshape({3, 4, 2});
	MockEdge<double> edge(
		teq::TensptrT(new MockTensor(teq::Shape({3, 1, 2}))),
		std::vector<double>{2, 8, 4, 5, 6, 7},
		outshape,
		std::vector<double>{1, 4});
	auto r = eigen::extend(edge);

	double* raw = r->get_ptr();
	r->assign();

	std::vector<double> expect_raw = {
		2, 8, 4, 2, 8, 4, 2, 8, 4, 2, 8, 4,
		5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7,
	};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_ARREQ(expect_raw, got_raw);
}


TEST(OPERATOR, Permute)
{
	{
		teq::Shape outshape({3, 2, 2});
		MockEdge<double> edge(
			teq::TensptrT(new MockTensor(teq::Shape({2, 2, 3}))),
			std::vector<double>{2, 8, 4, 5, 6, 7, 1, 0, 9, 11, 10, 12},
			outshape,
			std::vector<double>{2, 0, 1, 3, 4, 5, 6, 7});
		auto r = eigen::permute(edge);

		double* raw = r->get_ptr();
		r->assign();

		std::vector<double> expect_raw = {
			2, 6, 9, 8, 7, 11,
			4, 1, 10, 5, 0, 12,
		};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_ARREQ(expect_raw, got_raw);
	}
	{
		teq::Shape outshape({3, 2});
		MockEdge<double> edge(
			teq::TensptrT(new MockTensor(teq::Shape({2, 3}))),
			std::vector<double>{2, 8, 4, 5, 6, 7},
			outshape,
			std::vector<double>{1, 0, 2, 3, 4, 5, 6, 7});
		auto r = eigen::permute(edge);

		double* raw = r->get_ptr();
		r->assign();

		std::vector<double> expect_raw = {2, 4, 6, 8, 5, 7};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_ARREQ(expect_raw, got_raw);
	}
}


TEST(OPERATOR, Reshape)
{
	teq::Shape outshape({1, 3, 2});
	MockEdge<double> edge(
		teq::TensptrT(new MockTensor(teq::Shape({2, 1, 3}))),
		std::vector<double>{2, 8, 4, 5, 6, 7},
		outshape);
	auto r = eigen::reshape(edge);

	double* raw = r->get_ptr();
	r->assign();

	std::vector<double> expect_raw = {
		2, 8, 4, 5, 6, 7,
	};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_ARREQ(expect_raw, got_raw);
}


TEST(OPERATOR, Slice)
{
	teq::Shape outshape({2, 1});
	MockEdge<double> edge(
		teq::TensptrT(new MockTensor(teq::Shape({3, 2}))),
		std::vector<double>{2, 8, 4, 5, 6, 7},
		outshape, std::vector<double>{1, 2, 1, 1});
	auto r = eigen::slice(edge);

	double* raw = r->get_ptr();
	r->assign();

	std::vector<double> expect_raw = {6, 7};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_ARREQ(expect_raw, got_raw);
}


TEST(OPERATOR, GroupConcat)
{
	teq::Shape outshape({2, 4});
	MockEdge<double> edgea(
		teq::TensptrT(new MockTensor(teq::Shape({1, 4}))),
		std::vector<double>{2, 8, 4, 5}, outshape,
		std::vector<double>{0});
	MockEdge<double> edgeb(
		teq::TensptrT(new MockTensor(teq::Shape({1, 4}))),
		std::vector<double>{1, 0, 3, 9}, outshape);
	auto r = eigen::group_concat<double>({edgea, edgeb});

	double* raw = r->get_ptr();
	r->assign();

	std::vector<double> expect_raw = {2, 1, 8, 0, 4, 3, 5, 9};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_ARREQ(expect_raw, got_raw);
}


TEST(OPERATOR, GroupSum)
{
	teq::Shape outshape({2, 3});
	MockEdge<double> edgea(
		teq::TensptrT(new MockTensor(teq::Shape({2, 3}))),
		std::vector<double>{2, 8, 4, 5, 6, 7}, outshape);
	MockEdge<double> edgeb(
		teq::TensptrT(new MockTensor(teq::Shape({2, 3}))),
		std::vector<double>{1, 0, 3, 9, 10, 11}, outshape);
	MockEdge<double> edgec(
		teq::TensptrT(new MockTensor(teq::Shape({2, 3}))),
		std::vector<double>{4.2, 1, 7.1, 1, 2, 1.1}, outshape);
	auto r = eigen::group_sum<double>({edgea, edgeb, edgec});

	double* raw = r->get_ptr();
	r->assign();

	std::vector<double> expect_raw = {7.2, 9, 14.1, 15, 18, 19.1};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_ARREQ(expect_raw, got_raw);
}


TEST(OPERATOR, GroupProd)
{
	teq::Shape outshape({2, 3});
	MockEdge<double> edgea(
		teq::TensptrT(new MockTensor(teq::Shape({2, 3}))),
		std::vector<double>{2, 8, 4, 5, 6, 7}, outshape);
	MockEdge<double> edgeb(
		teq::TensptrT(new MockTensor(teq::Shape({2, 3}))),
		std::vector<double>{1, 0, 3, 9, 10, 11}, outshape);
	MockEdge<double> edgec(
		teq::TensptrT(new MockTensor(teq::Shape({2, 3}))),
		std::vector<double>{4, 1, 7, 1, 2, 1}, outshape);
	auto r = eigen::group_prod<double>({edgea, edgeb, edgec});

	double* raw = r->get_ptr();
	r->assign();

	std::vector<double> expect_raw = {8, 0, 84, 45, 120, 77};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_ARREQ(expect_raw, got_raw);
}


TEST(OPERATOR, Pad)
{
	teq::Shape outshape({4, 3});
	MockEdge<double> edge(
		teq::TensptrT(new MockTensor(teq::Shape({2, 3}))),
		std::vector<double>{2, 8, 4, 5, 6, 7},
		outshape, std::vector<double>{1, 1});
	auto r = eigen::pad(edge);

	double* raw = r->get_ptr();
	r->assign();

	std::vector<double> expect_raw = {
		0, 2, 8, 0,
		0, 4, 5, 0,
		0, 6, 7, 0,
	};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_ARREQ(expect_raw, got_raw);
}


TEST(OPERATOR, Stride)
{
	teq::Shape outshape({2, 2});
	MockEdge<double> edge(
		teq::TensptrT(new MockTensor(teq::Shape({2, 3}))),
		std::vector<double>{2, 8, 4, 5, 6, 7},
		outshape, std::vector<double>{1, 2});
	auto r = eigen::stride(edge);

	double* raw = r->get_ptr();
	r->assign();

	std::vector<double> expect_raw = {
		2, 8, 6, 7,
	};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_ARREQ(expect_raw, got_raw);
}


TEST(OPERATOR, Scatter)
{
	teq::Shape outshape({3, 3});
	MockEdge<double> edge(
		teq::TensptrT(new MockTensor(teq::Shape({2, 2}))),
		std::vector<double>{2, 8, 4, 5},
		outshape, std::vector<double>{2, 2});
	auto r = eigen::scatter(edge);

	double* raw = r->get_ptr();
	r->assign();

	std::vector<double> expect_raw = {
		2, 0, 8,
		0, 0, 0,
		4, 0, 5,
	};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_ARREQ(expect_raw, got_raw);
}


TEST(OPERATOR, Reverse)
{
	teq::Shape outshape({2, 3});
	MockEdge<double> edge(
		teq::TensptrT(new MockTensor(outshape)),
		std::vector<double>{2, 8, 4, 5, 6, 7},
		outshape, std::vector<double>{1});
	auto r = eigen::reverse(edge);

	double* raw = r->get_ptr();
	r->assign();

	std::vector<double> expect_raw = {
		6, 7, 4, 5, 2, 8
	};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_ARREQ(expect_raw, got_raw);
}


TEST(OPERATOR, Concat)
{
	teq::Shape outshape({3, 3});
	MockEdge<double> edgea(
		teq::TensptrT(new MockTensor(teq::Shape({2, 3}))),
		std::vector<double>{2, 8, 4, 5, 7, 6}, outshape,
		std::vector<double>{0});
	MockEdge<double> edgeb(
		teq::TensptrT(new MockTensor(teq::Shape({1, 3}))),
		std::vector<double>{1, 0, 3}, outshape);
	auto r = eigen::concat(edgea, edgeb);

	double* raw = r->get_ptr();
	r->assign();

	std::vector<double> expect_raw = {
		2, 8, 1,
		4, 5, 0,
		7, 6, 3,
	};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_ARREQ(expect_raw, got_raw);
}


TEST(OPERATOR, Convolution)
{
	teq::Shape outshape({3, 2});
	MockEdge<double> image(
		teq::TensptrT(new MockTensor(teq::Shape({3, 3}))),
		std::vector<double>{
			2, 8, 4,
			5, 7, 6,
			9, 1, 0,
		},
		outshape);
	MockEdge<double> kernel(
		teq::TensptrT(new MockTensor(teq::Shape({2}))),
		std::vector<double>{0.3, 0.6}, outshape,
		std::vector<double>{1});
	auto r = eigen::convolution(image, kernel);
	double* raw = r->get_ptr();
	r->assign();

	std::vector<double> expect_raw = {
		2 * 0.3 + 5 * 0.6, 8 * 0.3 + 7 * 0.6, 4 * 0.3 + 6 * 0.6,
		5 * 0.3 + 9 * 0.6, 7 * 0.3 + 1 * 0.6, 6 * 0.3 + 0 * 0.6,
	};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_ARREQ(expect_raw, got_raw);
}


TEST(INTERNAL, DecodePair)
{
	std::vector<double> a = {1, 2, 3, 4, 5, 6};
	std::vector<double> bad = {1, 2, 3, 4, 5};

	auto apairs = eigen::decode_pair<size_t>(a);
	EXPECT_EQ(3, apairs.size());
	for (size_t i = 0; i < 3; ++i)
	{
		auto apair = apairs[i];
		EXPECT_EQ(apair.first + 1, apair.second);
		EXPECT_EQ(i * 2 + 1, apair.first);
	}
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

	EXPECT_ARREQ(values, gotb);
	EXPECT_ARREQ(values, gotc);
	EXPECT_ARREQ(values, gotd);

	eigen::MatrixT<double> m(3, 3);
	std::copy(values.begin(), values.end(), m.data());

	auto e = eigen::mat_to_matmap(m);

	auto f = eigen::mat_to_tensmap(m);

	auto rawe = e.data();
	auto rawf = f.data();

	std::vector<double> gote(rawe, rawe + 9);
	std::vector<double> gotf(rawf, rawf + 9);

	EXPECT_ARREQ(values, gote);
	EXPECT_ARREQ(values, gotf);
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
	EXPECT_ARREQ(expect, got_mat);
	EXPECT_ARREQ(expect, got_mten);

	std::vector<double> expect_static = {1, 2, 3, 4, 5, 6};
	EXPECT_ARREQ(expect_static, got_tens);

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
	EXPECT_ARREQ(coord, c);

	MockEdge<double> badedge(
		teq::TensptrT(new MockTensor(teq::Shape(shape))),
		std::vector<double>{2, 8, 4, 5, 7, 6}, shape);
	EXPECT_FATAL(get_coorder(badedge), "coorder not found");

	JunkEdge<double> worstedge;
	EXPECT_FATAL(get_coorder(worstedge), "cannot find array coorder");
}


#endif // DISABLE_OPERATOR_TEST
