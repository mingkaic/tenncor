
#ifndef DISABLE_QUERY_ATTRS_TEST


#include "gtest/gtest.h"

#include "dbg/print/teq.hpp"

#include "internal/teq/mock/mock.hpp"

#include "internal/query/querier.hpp"
#include "internal/query/parse.hpp"


using ::testing::_;
using ::testing::Return;
using ::testing::Const;


TEST(ATTRS, FindByAttrTensKeyDirectSubgraph)
{
	auto a = make_var(teq::Shape(), "A");
	auto b = make_var(teq::Shape(), "B");
	auto c = make_var(teq::Shape(), "C");

	auto d = make_fnc("SIN", 0, teq::TensptrsT{b});
	auto f = make_fnc("SUB", 1, teq::TensptrsT{a, c});
	auto root = make_fnc("SUB", 1, teq::TensptrsT{f, d});

	teq::TensorObj tensobjc(c);
	marsh::Number<double> numobj(333.4);
	teq::TensorObj tensobjf(f);

	EXPECT_CALL(*f, size()).WillRepeatedly(Return(1));
	EXPECT_CALL(*f, ls_attrs()).WillRepeatedly(Return(types::StringsT{"yodoo"}));
	EXPECT_CALL(*f, get_attr("yodoo")).WillRepeatedly(Return(&tensobjc));
	EXPECT_CALL(Const(*f), get_attr("yodoo")).WillRepeatedly(Return(&tensobjc));
	EXPECT_CALL(*root, size()).WillRepeatedly(Return(2));
	EXPECT_CALL(*root, ls_attrs()).WillRepeatedly(Return(types::StringsT{"numbers","tensors"}));
	EXPECT_CALL(*root, get_attr("numbers")).WillRepeatedly(Return(&numobj));
	EXPECT_CALL(*root, get_attr("tensors")).WillRepeatedly(Return(&tensobjf));
	EXPECT_CALL(Const(*root), get_attr("numbers")).WillRepeatedly(Return(&numobj));
	EXPECT_CALL(Const(*root), get_attr("tensors")).WillRepeatedly(Return(&tensobjf));

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
	query::json_parse(cond, condjson);
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
	auto a = make_var(teq::Shape(), "A");
	auto b = make_var(teq::Shape(), "B");
	auto c = make_var(teq::Shape(), "C");

	auto d = make_fnc("SIN", 0, teq::TensptrsT{b});
	auto f = make_fnc("SUB", 1, teq::TensptrsT{a, c});
	auto root = make_fnc("SUB", 1, teq::TensptrsT{f, d});

	teq::LayerObj layrobj("funky", c);
	marsh::Number<double> numobj(333.4);
	teq::TensorObj tensobj(f);

	EXPECT_CALL(*f, size()).WillRepeatedly(Return(1));
	EXPECT_CALL(*f, ls_attrs()).WillRepeatedly(Return(types::StringsT{"yodoo"}));
	EXPECT_CALL(*f, get_attr("yodoo")).WillRepeatedly(Return(&layrobj));
	EXPECT_CALL(Const(*f), get_attr("yodoo")).WillRepeatedly(Return(&layrobj));
	EXPECT_CALL(*root, size()).WillRepeatedly(Return(2));
	EXPECT_CALL(*root, ls_attrs()).WillRepeatedly(Return(types::StringsT{"numbers","yodoo"}));
	EXPECT_CALL(*root, get_attr("numbers")).WillRepeatedly(Return(&numobj));
	EXPECT_CALL(*root, get_attr("yodoo")).WillRepeatedly(Return(&tensobj));
	EXPECT_CALL(Const(*root), get_attr("numbers")).WillRepeatedly(Return(&numobj));
	EXPECT_CALL(Const(*root), get_attr("yodoo")).WillRepeatedly(Return(&tensobj));

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
	query::json_parse(cond, condjson);
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
	auto a = make_var(teq::Shape(), "A");
	auto b = make_var(teq::Shape(), "B");
	auto c = make_var(teq::Shape(), "C");

	auto d = make_fnc("SIN", 0, teq::TensptrsT{b});
	auto f = make_fnc("SUB", 1, teq::TensptrsT{a, c});
	auto root = make_fnc("SUB", 1, teq::TensptrsT{f, d});

	teq::LayerObj layrobj("funky", c);
	marsh::Number<double> numobj(333.4);
	teq::LayerObj layrobj2("punky", f);

	EXPECT_CALL(*f, size()).WillRepeatedly(Return(1));
	EXPECT_CALL(*f, ls_attrs()).WillRepeatedly(Return(types::StringsT{"yodoo"}));
	EXPECT_CALL(*f, get_attr("yodoo")).WillRepeatedly(Return(&layrobj));
	EXPECT_CALL(Const(*f), get_attr("yodoo")).WillRepeatedly(Return(&layrobj));
	EXPECT_CALL(*root, size()).WillRepeatedly(Return(2));
	EXPECT_CALL(*root, ls_attrs()).WillRepeatedly(Return(types::StringsT{"numbers","yodoo"}));
	EXPECT_CALL(*root, get_attr("numbers")).WillRepeatedly(Return(&numobj));
	EXPECT_CALL(*root, get_attr("yodoo")).WillRepeatedly(Return(&layrobj2));
	EXPECT_CALL(Const(*root), get_attr("numbers")).WillRepeatedly(Return(&numobj));
	EXPECT_CALL(Const(*root), get_attr("yodoo")).WillRepeatedly(Return(&layrobj2));

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
	query::json_parse(cond, condjson);
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
	auto a = make_var(teq::Shape(), "A");
	auto b = make_var(teq::Shape(), "B");
	auto c = make_var(teq::Shape(), "C");

	auto d = make_fnc("SUB", 1, teq::TensptrsT{b, c});
	auto f = make_fnc("SUB", 1, teq::TensptrsT{a, d});
	auto root = make_fnc("SUB", 1, teq::TensptrsT{f, d});

	teq::TensorObj tensobja(a);
	teq::TensorObj tensobjd(d);

	EXPECT_CALL(*d, size()).WillRepeatedly(Return(1));
	EXPECT_CALL(*d, ls_attrs()).WillRepeatedly(Return(types::StringsT{"yodoo"}));
	EXPECT_CALL(*d, get_attr("yodoo")).WillRepeatedly(Return(&tensobja));
	EXPECT_CALL(Const(*d), get_attr("yodoo")).WillRepeatedly(Return(&tensobja));
	EXPECT_CALL(*f, size()).WillRepeatedly(Return(1));
	EXPECT_CALL(*f, ls_attrs()).WillRepeatedly(Return(types::StringsT{"yodoo"}));
	EXPECT_CALL(*f, get_attr("yodoo")).WillRepeatedly(Return(&tensobjd));
	EXPECT_CALL(Const(*f), get_attr("yodoo")).WillRepeatedly(Return(&tensobjd));

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
	query::json_parse(cond, condjson);
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
	auto a = make_var(teq::Shape(), "A");
	auto b = make_var(teq::Shape(), "B");
	auto c = make_var(teq::Shape(), "C");

	auto d = make_fnc("SIN", 0, teq::TensptrsT{b});
	auto f = make_fnc("SUB", 1, teq::TensptrsT{a, c});
	auto root = make_fnc("SUB", 1, teq::TensptrsT{a, d});

	teq::TensorObj tensobjc(c);
	marsh::Number<double> numobj(333.4);
	teq::TensorObj tensobjf(f);

	EXPECT_CALL(*f, size()).WillRepeatedly(Return(1));
	EXPECT_CALL(*f, ls_attrs()).WillRepeatedly(Return(types::StringsT{"yodoo"}));
	EXPECT_CALL(*f, get_attr("yodoo")).WillRepeatedly(Return(&tensobjc));
	EXPECT_CALL(Const(*f), get_attr("yodoo")).WillRepeatedly(Return(&tensobjc));
	EXPECT_CALL(*root, size()).WillRepeatedly(Return(2));
	EXPECT_CALL(*root, ls_attrs()).WillRepeatedly(Return(types::StringsT{"numbers","tensors"}));
	EXPECT_CALL(*root, get_attr("numbers")).WillRepeatedly(Return(&numobj));
	EXPECT_CALL(*root, get_attr("tensors")).WillRepeatedly(Return(&tensobjf));
	EXPECT_CALL(Const(*root), get_attr("numbers")).WillRepeatedly(Return(&numobj));
	EXPECT_CALL(Const(*root), get_attr("tensors")).WillRepeatedly(Return(&tensobjf));

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
	query::json_parse(cond, condjson);
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


#endif // DISABLE_QUERY_ATTRS_TEST
