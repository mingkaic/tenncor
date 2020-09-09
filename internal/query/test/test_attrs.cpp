
#ifndef DISABLE_ATTRS_TEST


#include "gtest/gtest.h"

#include "dbg/print/teq.hpp"

#include "internal/teq/mock/leaf.hpp"
#include "internal/teq/mock/functor.hpp"

#include "internal/query/querier.hpp"
#include "internal/query/parse.hpp"


TEST(ATTRS, FindByAttrTensKeyDirectSubgraph)
{
	teq::TensptrT a(new MockLeaf(teq::Shape(), "A"));
	teq::TensptrT b(new MockLeaf(teq::Shape(), "B"));
	teq::TensptrT c(new MockLeaf(teq::Shape(), "C"));

	auto df = new MockFunctor(teq::TensptrsT{b}, teq::Opcode{"SIN", 0});
	teq::TensptrT d(df);

	auto ff = new MockFunctor(teq::TensptrsT{a, c}, teq::Opcode{"SUB", 1});
	ff->add_attr("yodoo", std::make_unique<teq::TensorObj>(c));
	teq::TensptrT f(ff);

	auto gf = new MockFunctor(teq::TensptrsT{f, d}, teq::Opcode{"SUB", 1});
	gf->add_attr("numbers", std::make_unique<marsh::Number<double>>(333.4));
	gf->add_attr("tensors", std::make_unique<teq::TensorObj>(f));
	teq::TensptrT root(gf);

	std::stringstream condjson;
	condjson <<
		"{\"op\":{"
			"\"opname\":\"SUB\","
			"\"attrs\":{"
				"\"yodoo\":{"
					"\"node\":{\"symb\":\"C\"}"
				"}"
			"},"
			"\"args\":[{\"symb\":\"A\"},{\"symb\":\"B\"}]"
		"}}";
	query::Node cond;
	json_parse(cond, condjson);
	query::Query matcher;
	root->accept(matcher);
	auto detections = matcher.match(cond);
	teq::TensSetT roots(detections.begin(), detections.end());

	ASSERT_EQ(1, roots.size());
	char expected[] =
		"(SUB)\n"
		"_`--(constant:A)\n"
		"_`--(constant:C)\n";

	PrettyEquation peq;
	std::stringstream ss;
	peq.print(ss, *roots.begin());
	EXPECT_STREQ(expected, ss.str().c_str());
}


TEST(ATTRS, FindByAttrTensKeyLayer)
{
	teq::TensptrT a(new MockLeaf(teq::Shape(), "A"));
	teq::TensptrT b(new MockLeaf(teq::Shape(), "B"));
	teq::TensptrT c(new MockLeaf(teq::Shape(), "C"));

	auto df = new MockFunctor(teq::TensptrsT{b}, teq::Opcode{"SIN", 0});
	teq::TensptrT d(df);

	auto ff = new MockFunctor(teq::TensptrsT{a, c}, teq::Opcode{"SUB", 1});
	ff->add_attr("yodoo", std::make_unique<teq::LayerObj>("funky", c));
	teq::TensptrT f(ff);

	auto gf = new MockFunctor(teq::TensptrsT{f, d}, teq::Opcode{"SUB", 1});
	gf->add_attr("numbers", std::make_unique<marsh::Number<double>>(333.4));
	gf->add_attr("yodoo", std::make_unique<teq::TensorObj>(f));
	teq::TensptrT root(gf);

	std::stringstream condjson;
	condjson <<
		"{\"op\":{"
			"\"opname\":\"SUB\","
			"\"attrs\":{"
				"\"yodoo\":{"
					"\"layer\":{"
						"\"name\":\"funky\","
						"\"input\":{\"symb\":\"C\"}"
					"}"
				"}"
			"},"
			"\"args\":[{\"symb\":\"A\"},{\"symb\":\"B\"}]"
		"}}";
	query::Node cond;
	json_parse(cond, condjson);
	query::Query matcher;
	root->accept(matcher);
	auto detections = matcher.match(cond);
	teq::TensSetT roots(detections.begin(), detections.end());

	ASSERT_EQ(1, roots.size());
	char expected[] =
		"(SUB)\n"
		"_`--(constant:A)\n"
		"_`--(constant:C)\n";

	PrettyEquation peq;
	std::stringstream ss;
	peq.print(ss, *roots.begin());
	EXPECT_STREQ(expected, ss.str().c_str());
}


TEST(ATTRS, FindByAttrLayerKeyName)
{
	teq::TensptrT a(new MockLeaf(teq::Shape(), "A"));
	teq::TensptrT b(new MockLeaf(teq::Shape(), "B"));
	teq::TensptrT c(new MockLeaf(teq::Shape(), "C"));

	auto df = new MockFunctor(teq::TensptrsT{b}, teq::Opcode{"SIN", 0});
	teq::TensptrT d(df);

	auto ff = new MockFunctor(teq::TensptrsT{a, c}, teq::Opcode{"SUB", 1});
	ff->add_attr("yodoo", std::make_unique<teq::LayerObj>("funky", c));
	teq::TensptrT f(ff);

	auto gf = new MockFunctor(teq::TensptrsT{f, d}, teq::Opcode{"SUB", 1});
	gf->add_attr("numbers", std::make_unique<marsh::Number<double>>(333.4));
	gf->add_attr("yodoo", std::make_unique<teq::LayerObj>("punky", f));
	teq::TensptrT root(gf);

	std::stringstream condjson;
	condjson <<
		"{\"op\":{"
			"\"opname\":\"SUB\","
			"\"attrs\":{"
				"\"yodoo\":{"
					"\"layer\":{"
						"\"name\":\"funky\","
						"\"input\":{\"symb\":\"C\"}"
					"}"
				"}"
			"},"
			"\"args\":[{\"symb\":\"A\"},{\"symb\":\"B\"}]"
		"}}";
	query::Node cond;
	json_parse(cond, condjson);
	query::Query matcher;
	root->accept(matcher);
	auto detections = matcher.match(cond);
	teq::TensSetT roots(detections.begin(), detections.end());

	ASSERT_EQ(1, roots.size());
	char expected[] =
		"(SUB)\n"
		"_`--(constant:A)\n"
		"_`--(constant:C)\n";

	PrettyEquation peq;
	std::stringstream ss;
	peq.print(ss, *roots.begin());
	EXPECT_STREQ(expected, ss.str().c_str());
}


TEST(ATTRS, FindByAttrValDirectSubgraph)
{
	teq::TensptrT a(new MockLeaf(teq::Shape(), "A"));
	teq::TensptrT b(new MockLeaf(teq::Shape(), "B"));
	teq::TensptrT c(new MockLeaf(teq::Shape(), "C"));

	auto df = new MockFunctor(teq::TensptrsT{b, c}, teq::Opcode{"SUB", 1});
	df->add_attr("yodoo", std::make_unique<teq::TensorObj>(a));
	teq::TensptrT d(df);

	auto ff = new MockFunctor(teq::TensptrsT{a, d}, teq::Opcode{"SUB", 1});
	ff->add_attr("yodoo", std::make_unique<teq::TensorObj>(d));
	teq::TensptrT f(ff);

	auto gf = new MockFunctor(teq::TensptrsT{f, d}, teq::Opcode{"SUB", 1});
	teq::TensptrT root(gf);

	std::stringstream condjson;
	condjson <<
		"{"
			"\"op\":{"
				"\"opname\":\"SUB\","
				"\"args\":["
					"{\"symb\":\"A\"},"
					"{"
						"\"op\":{"
							"\"opname\":\"SUB\","
							"\"attrs\":{"
								"\"yodoo\":{"
									"\"node\":{\"symb\":\"A\"}"
								"}"
							"},"
							"\"args\":["
								"{\"symb\":\"B\"},"
								"{\"symb\":\"C\"}"
							"]"
						"}"
					"}"
				"]"
			"}"
		"}";
	query::Node cond;
	json_parse(cond, condjson);
	query::Query matcher;
	root->accept(matcher);
	auto detections = matcher.match(cond);
	teq::TensSetT roots(detections.begin(), detections.end());

	ASSERT_EQ(1, roots.size());
	char expected[] =
		"(SUB)\n"
		"_`--(constant:A)\n"
		"_`--(SUB)\n"
		"_____`--(constant:B)\n"
		"_____`--(constant:C)\n";

	PrettyEquation peq;
	std::stringstream ss;
	peq.print(ss, *roots.begin());
	EXPECT_STREQ(expected, ss.str().c_str());
}


TEST(ATTRS, FindByAttrTensKeyUnderAttrSubgraph)
{
	teq::TensptrT a(new MockLeaf(teq::Shape(), "A"));
	teq::TensptrT b(new MockLeaf(teq::Shape(), "B"));
	teq::TensptrT c(new MockLeaf(teq::Shape(), "C"));

	auto df = new MockFunctor(teq::TensptrsT{b}, teq::Opcode{"SIN", 0});
	teq::TensptrT d(df);

	auto ff = new MockFunctor(teq::TensptrsT{a, c}, teq::Opcode{"SUB", 1});
	ff->add_attr("yodoo", std::make_unique<teq::TensorObj>(c));
	teq::TensptrT f(ff);

	auto gf = new MockFunctor(teq::TensptrsT{a, d}, teq::Opcode{"SUB", 1});
	gf->add_attr("numbers", std::make_unique<marsh::Number<double>>(333.4));
	gf->add_attr("tensors", std::make_unique<teq::TensorObj>(f));
	teq::TensptrT root(gf);

	std::stringstream condjson;
	condjson <<
		"{\"op\":{"
			"\"opname\":\"SUB\","
			"\"attrs\":{"
				"\"yodoo\":{"
					"\"node\":{\"symb\":\"C\"}"
				"}"
			"},"
			"\"args\":[{\"symb\":\"A\"},{\"symb\":\"B\"}]"
		"}}";
	query::Node cond;
	json_parse(cond, condjson);
	query::Query matcher;
	root->accept(matcher);
	auto detections = matcher.match(cond);
	teq::TensSetT roots(detections.begin(), detections.end());

	ASSERT_EQ(1, roots.size());
	char expected[] =
		"(SUB)\n"
		"_`--(constant:A)\n"
		"_`--(constant:C)\n";

	PrettyEquation peq;
	std::stringstream ss;
	peq.print(ss, *roots.begin());
	EXPECT_STREQ(expected, ss.str().c_str());
}


#endif // DISABLE_ATTRS_TEST
