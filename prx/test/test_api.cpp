
#ifndef DISABLE_API_TEST


#include "gtest/gtest.h"

#include "testutil/common.hpp"

#include "ead/session.hpp"
#include "ead/constant.hpp"
#include "ead/grader.hpp"

#include "prx/api.hpp"

#include "dbg/tensor.hpp"


TEST(API, Conv2d)
{
	ade::Shape expected_shape({4, 2, 2, 3});
	std::vector<float> exdata = {
		2.58359122276306152, 2.03315830230712891, 2.43214130401611328,
		2.69100403785705566, 2.10950589179992676, 1.7037736177444458,

		2.17994093894958496, 2.39612627029418945, 2.91597318649291992,
		2.27325201034545898, 3.04842686653137207, 3.21160554885864258,


		2.8041069507598877, 2.36647367477416992, 2.75740742683410645,
		3.13139677047729492, 2.19258379936218262, 1.53170442581176758,

		2.83604764938354492, 2.64835643768310547, 2.61888360977172852,
		1.97365725040435791, 2.29979562759399414, 2.54902052879333496,


		1.75394785404205322, 1.47045016288757324, 2.16185975074768066,
		2.23173284530639648, 1.9423755407333374, 1.76172828674316406,

		1.80572724342346191, 2.23552799224853516, 2.78410840034484863,
		1.80081379413604736, 2.97679567337036133, 2.79419875144958496,


		2.1808316707611084, 2.27519059181213379, 1.94113290309906006,
		2.74104166030883789, 1.96674573421478271, 1.64521610736846924,

		1.89129197597503662, 2.2983555793762207, 1.55121612548828125,
		1.54055297374725342, 1.99850225448608398, 1.96307921409606934,
	};

	std::vector<float> gimage_data = {
		1.09593232939999985, 1.97934496279999994, 3.61954617560000003,
		4.8062426855, 2.52361384619999995, 2.82689772270000006,
		2.68998046159999982, 2.93699984289999971, 7.57487191919999958,

		8.07350438249999947, 4.88489145760000021, 5.13650453959999975,
		1.59404813220000019, 0.957654880099999994, 3.95532574360000044,
		3.26726169699999991, 2.36127761139999981, 2.30960681690000014,

		1.09593232939999985, 1.97934496279999994, 3.61954617560000003,
		4.8062426855, 2.52361384619999995, 2.82689772270000006,
		2.68998046159999982, 2.93699984289999971, 7.57487191919999958,


		8.07350438249999947, 4.88489145760000021, 5.13650453959999975,
		1.59404813220000019, 0.957654880099999994, 3.95532574360000044,
		3.26726169699999991, 2.36127761139999981, 2.30960681690000014,

		1.09593232939999985, 1.97934496279999994, 3.61954617560000003,
		4.8062426855, 2.52361384619999995, 2.82689772270000006,
		2.68998046159999982, 2.93699984289999971, 7.57487191919999958,

		8.07350438249999947, 4.88489145760000021, 5.13650453959999975,
		1.59404813220000019, 0.957654880099999994, 3.95532574360000044,
		3.26726169699999991, 2.36127761139999981, 2.30960681690000014,
	};

	std::vector<float> gkernel_data = {
		7.64971495459999851, 7.64971495459999851,
		7.64971495459999851, 7.64971495459999851,

		6.8669139211000001, 6.8669139211000001,
		6.8669139211000001, 6.8669139211000001,


		7.2851931024999983, 7.2851931024999983,
		7.2851931024999983, 7.2851931024999983,

		7.17586653469999991, 7.17586653469999991,
		7.17586653469999991, 7.17586653469999991,


		6.62693156719999976, 6.62693156719999976,
		6.62693156719999976, 6.62693156719999976,

		6.84776966030000001, 6.84776966030000001,
		6.84776966030000001, 6.84776966030000001,


		6.36579149010000034, 6.36579149010000034,
		6.36579149010000034, 6.36579149010000034,

		7.01482652649999938, 7.01482652649999938,
		7.01482652649999938, 7.01482652649999938,
	};

	// in, width, height, batch
	ade::Shape shape({2, 3, 3, 3});
	std::vector<float> data = {
		0.5145785303, 0.9409955438, 0.1350188042,
		0.2677016750, 0.4020208487, 0.5040458798,
		0.8192468218, 0.7658994366, 0.9443380545,

		0.8428559589, 0.6748943724, 0.5914301605,
		0.5280673215, 0.3521708357, 0.7903877913,
		0.5300812313, 0.5143651515, 0.8654963056,

		0.1639687924, 0.1108488638, 0.9085991291,
		0.5425003410, 0.3953019238, 0.7956724151,
		0.3879704012, 0.2140292525, 0.8198060252,


		0.9628650582, 0.6291796215, 0.1776727903,
		0.0040358712, 0.9880141413, 0.3916120731,
		0.0437538110, 0.2048394146, 0.7365598359,

		0.6773088120, 0.1327919586, 0.7029628276,
		0.9971195524, 0.6911652047, 0.1615525364,
		0.6025299897, 0.7520521216, 0.9733867666,

		0.3372541587, 0.0085195242, 0.9951960084,
		0.0986719259, 0.7322653007, 0.2668785252,
		0.3265283538, 0.1475841700, 0.6051328539
	};

	// out, in, width, height
	ade::Shape kshape({4, 2, 2, 2});
	std::vector<float> kdata = {
		0.7274310793, 0.0569064784,
		0.2410496121, 0.0705451596,

		0.2806072993, 0.8342554197,
		0.1691792622, 0.6953029816,


		0.6696543218, 0.3976814420,
		0.6456448709, 0.8106332115,

		0.2097730182, 0.7364004705,
		0.9676746447, 0.9130495893,


		0.5560669713, 0.5356655997,
		0.0722204852, 0.4300950760,

		0.2673988657, 0.2106666978,
		0.1693800931, 0.3102092235,


		0.9783402914, 0.0155971183,
		0.9331012682, 0.4342389335,

		0.2543819707, 0.4189256744,
		0.8692180419, 0.7670811299
	};

	ead::NodeptrT<float> image = ead::make_constant<float>(data.data(), shape);
	ead::NodeptrT<float> kernel = ead::make_constant<float>(kdata.data(), kshape);
	ead::NodeptrT<float> bias = ead::make_constant_scalar<float>(0, ade::Shape({4}));

	ead::NodeptrT<float> out = prx::conv2d(image, kernel, bias);

	ead::Session<float> session;
	session.track(out->get_tensor().get());
	session.update();
	ade::Shape conved_shape = out->shape();
	ASSERT_ARREQ(expected_shape, conved_shape);
	ASSERT_EQ(exdata.size(), conved_shape.n_elems());

	float* conved_data = (float*) out->data();
	for (size_t i = 0, n = exdata.size(); i < n; ++i)
	{
		EXPECT_FLOAT_EQ(exdata[i], conved_data[i]);
	}

	ead::NodeptrT<float> gimage = ead::derive(out, image);
	ead::NodeptrT<float> gkernel = ead::derive(out, kernel);
	session.track(gimage->get_tensor().get());
	session.track(gkernel->get_tensor().get());
	session.update();

	{
		auto gotshape = gimage->shape();
		ASSERT_ARREQ(shape, gotshape);
		ASSERT_EQ(gimage_data.size(), gotshape.n_elems());
		float* goptr = (float*) gimage->data();
		for (size_t i = 0, n = gimage_data.size(); i < n; ++i)
		{
			EXPECT_FLOAT_EQ(gimage_data[i], goptr[i]);
		}
	}

	{
		auto gotshape = gkernel->shape();
		ASSERT_ARREQ(kshape, gotshape);
		ASSERT_EQ(gkernel_data.size(), gotshape.n_elems());
		float* goptr = (float*) gkernel->data();
		for (size_t i = 0, n = gkernel_data.size(); i < n; ++i)
		{
			EXPECT_FLOAT_EQ(gkernel_data[i], goptr[i]);
		}
	}
}


#endif // DISABLE_API_TEST
