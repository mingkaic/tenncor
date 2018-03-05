//
// Created by Mingkai Chen on 2016-08-29.
//

#ifndef DISABLE_TENSOR_MODULE_TESTS

#include <algorithm>
#include <numeric>
#include <limits>

#include "gtest/gtest.h"

#include "tests/utils/sgen.hpp"
#include "tests/utils/check.hpp"

#include "tensor/data_src.hpp"
#include "tensor/tensor.hpp"


#ifndef DISABLE_DSRC_TEST


class DATA_SRC : public testify::fuzz_test {};


using namespace testutils;


#define RAND_TYPE (TENS_TYPE) fuzzer->get_int(1, "type", {1, N_TYPE - 1})[0]

#define WITHIN_LIMITS(TYPE) {std::numeric_limits<TYPE>::min(), std::numeric_limits<TYPE>::max()}

#define RSCALAR(TYPE, OUT, EXPECT, FUZZ_F) \
TYPE OUT = FUZZ_F(1, "ci_" + std::string(#OUT) + "_" + std::string(#TYPE), WITHIN_LIMITS(TYPE))[0]; \
EXPECT = std::string(sizeof(TYPE), ' '); \
std::memcpy(&EXPECT[0], &OUT, sizeof(TYPE));

#define RVEC(TYPE, FUZZ_F) \
auto ovec = FUZZ_F(n_snippet, "ci_vec_" + std::string(#TYPE), WITHIN_LIMITS(TYPE)); \
std::vector<TYPE> vec(ovec.begin(), ovec.end()); \
expect = std::string(n_snippet * sizeof(TYPE), ' '); \
std::memcpy(&expect[0], &vec[0], n_snippet * sizeof(TYPE));

#define RMINMAX(TYPE, FUZZ_F) \
RSCALAR(TYPE, min, minstr, FUZZ_F) \
TYPE max = FUZZ_F(1, "ci_max_" + std::string(#TYPE), {min + 1, std::numeric_limits<TYPE>::max()})[0]; \
maxstr = std::string(sizeof(TYPE), ' '); \
std::memcpy(&maxstr[0], &max, sizeof(TYPE));

#define STRINGIFY(TYPE) \
std::vector<TYPE> vec = nnet::expose<TYPE>(&ten); \
everything = std::string(vec.size() * sizeof(TYPE), ' '); \
std::memcpy(&everything[0], &vec[0], vec.size() * sizeof(TYPE));


TENS_TYPE fuzz_const (testify::fuzz_test* fuzzer, 
	nnet::const_init& ci, std::string& expect)
{
	TENS_TYPE out = RAND_TYPE;
	switch (out)
	{
		case DOUBLE:
		{
			RSCALAR(double, scalar, expect, fuzzer->get_double)
			ci.set<double>(scalar);
		}
		break;
		case FLOAT:
		{
			RSCALAR(float, scalar, expect, fuzzer->get_double)
			ci.set<float>(scalar);
		}
		break;
		case INT8:
		{
			RSCALAR(int8_t, scalar, expect, fuzzer->get_int)
			ci.set<int8_t>(scalar);
		}
		break;
		case UINT8:
		{
			RSCALAR(uint8_t, scalar, expect, fuzzer->get_int)
			ci.set<uint8_t>(scalar);
		}
		break;
		case INT16:
		{
			RSCALAR(int16_t, scalar, expect, fuzzer->get_int)
			ci.set<int16_t>(scalar);
		}
		break;
		case UINT16:
		{
			RSCALAR(uint16_t, scalar, expect, fuzzer->get_int)
			ci.set<uint16_t>(scalar);
		}
		break;
		case INT32:
		{
			RSCALAR(int32_t, scalar, expect, fuzzer->get_int)
			ci.set<int32_t>(scalar);
		}
		break;
		case UINT32:
		{
			RSCALAR(uint32_t, scalar, expect, fuzzer->get_int)
			ci.set<uint32_t>(scalar);
		}
		break;
		case INT64:
		{
			RSCALAR(int64_t, scalar, expect, fuzzer->get_int)
			ci.set<int64_t>(scalar);
		}
		break;
		case UINT64:
		{
			RSCALAR(uint64_t, scalar, expect, fuzzer->get_int)
			ci.set<uint64_t>(scalar);
		}
		break;
		default:
			// will never happen
		break;
	}
	return out;
}


TENS_TYPE fuzz_vec (testify::fuzz_test* fuzzer, 
	nnet::const_init& ci, std::string& expect, size_t nelems)
{
	TENS_TYPE out = RAND_TYPE;
	size_t n_snippet = 1;
	if (nelems == 1)
	{
		n_snippet = fuzzer->get_int(1, "n_snippet", {1, nelems-1})[0];
	}
	switch (out)
	{
		case DOUBLE:
		{
			RVEC(double, fuzzer->get_double)
			ci.set<double>(vec);
		}
		break;
		case FLOAT:
		{
			RVEC(float, fuzzer->get_double)
			ci.set<float>(vec);
		}
		break;
		case INT8:
		{
			RVEC(int8_t, fuzzer->get_int)
			ci.set<int8_t>(vec);
		}
		break;
		case UINT8:
		{
			RVEC(uint8_t, fuzzer->get_int)
			ci.set<uint8_t>(vec);
		}
		break;
		case INT16:
		{
			RVEC(int16_t, fuzzer->get_int)
			ci.set<int16_t>(vec);
		}
		break;
		case UINT16:
		{
			RVEC(uint16_t, fuzzer->get_int)
			ci.set<uint16_t>(vec);
		}
		break;
		case INT32:
		{
			RVEC(int32_t, fuzzer->get_int)
			ci.set<int32_t>(vec);
		}
		break;
		case UINT32:
		{
			RVEC(uint32_t, fuzzer->get_int)
			ci.set<uint32_t>(vec);
		}
		break;
		case INT64:
		{
			RVEC(int64_t, fuzzer->get_int)
			ci.set<int64_t>(vec);
		}
		break;
		case UINT64:
		{
			RVEC(uint64_t, fuzzer->get_int)
			ci.set<uint64_t>(vec);
		}
		break;
		default:
			// will never happen
		break;
	}
	return out;
}


TENS_TYPE fuzz_uniform (testify::fuzz_test* fuzzer, 
	nnet::r_uniform_init& ui, std::string& minstr, std::string& maxstr)
{
	TENS_TYPE out = RAND_TYPE;
	switch (out)
	{
		case DOUBLE:
		{
			RMINMAX(double, fuzzer->get_double)
			ui.set<double>(min, max);
		}
		break;
		case FLOAT:
		{
			RMINMAX(float, fuzzer->get_double)
			ui.set<float>(min, max);
		}
		break;
		case INT8:
		{
			RMINMAX(int8_t, fuzzer->get_int)
			ui.set<int8_t>(min, max);
		}
		break;
		case UINT8:
		{
			RMINMAX(uint8_t, fuzzer->get_int)
			ui.set<uint8_t>(min, max);
		}
		break;
		case INT16:
		{
			RMINMAX(int16_t, fuzzer->get_int)
			ui.set<int16_t>(min, max);
		}
		break;
		case UINT16:
		{
			RMINMAX(uint16_t, fuzzer->get_int)
			ui.set<uint16_t>(min, max);
		}
		break;
		case INT32:
		{
			RMINMAX(int32_t, fuzzer->get_int)
			ui.set<int32_t>(min, max);
		}
		break;
		case UINT32:
		{
			RMINMAX(uint32_t, fuzzer->get_int)
			ui.set<uint32_t>(min, max);
		}
		break;
		case INT64:
		{
			RMINMAX(int64_t, fuzzer->get_int)
			ui.set<int64_t>(min, max);
		}
		break;
		case UINT64:
		{
			RMINMAX(uint64_t, fuzzer->get_int)
			ui.set<uint64_t>(min, max);
		}
		break;
		default:
			// will never happen
		break;
	}
	return out;
}


TENS_TYPE fuzz_normal (testify::fuzz_test* fuzzer, 
	nnet::r_normal_init& ni, std::string& meanstr, std::string& stdevstr)
{
	TENS_TYPE out = RAND_TYPE;
	switch (out)
	{
		case DOUBLE:
		{
			RSCALAR(double, mean, meanstr, fuzzer->get_double)
			RSCALAR(double, stdev, stdevstr, fuzzer->get_double)
			ni.set<double>(mean, stdev);
		}
		break;
		case FLOAT:
		{
			RSCALAR(float, mean, meanstr, fuzzer->get_double)
			RSCALAR(float, stdev, stdevstr, fuzzer->get_double)
			ni.set<float>(mean, stdev);
		}
		break;
		case INT8:
		{
			RSCALAR(int8_t, mean, meanstr, fuzzer->get_int)
			RSCALAR(int8_t, stdev, stdevstr, fuzzer->get_int)
			ni.set<int8_t>(mean, stdev);
		}
		break;
		case UINT8:
		{
			RSCALAR(uint8_t, mean, meanstr, fuzzer->get_int)
			RSCALAR(uint8_t, stdev, stdevstr, fuzzer->get_int)
			ni.set<uint8_t>(mean, stdev);
		}
		break;
		case INT16:
		{
			RSCALAR(int16_t, mean, meanstr, fuzzer->get_int)
			RSCALAR(int16_t, stdev, stdevstr, fuzzer->get_int)
			ni.set<int16_t>(mean, stdev);
		}
		break;
		case UINT16:
		{
			RSCALAR(uint16_t, mean, meanstr, fuzzer->get_int)
			RSCALAR(uint16_t, stdev, stdevstr, fuzzer->get_int)
			ni.set<uint16_t>(mean, stdev);
		}
		break;
		case INT32:
		{
			RSCALAR(int32_t, mean, meanstr, fuzzer->get_int)
			RSCALAR(int32_t, stdev, stdevstr, fuzzer->get_int)
			ni.set<int32_t>(mean, stdev);
		}
		break;
		case UINT32:
		{
			RSCALAR(uint32_t, mean, meanstr, fuzzer->get_int)
			RSCALAR(uint32_t, stdev, stdevstr, fuzzer->get_int)
			ni.set<uint32_t>(mean, stdev);
		}
		break;
		case INT64:
		{
			RSCALAR(int64_t, mean, meanstr, fuzzer->get_int)
			RSCALAR(int64_t, stdev, stdevstr, fuzzer->get_int)
			ni.set<int64_t>(mean, stdev);
		}
		break;
		case UINT64:
		{
			RSCALAR(uint64_t, mean, meanstr, fuzzer->get_int)
			RSCALAR(uint64_t, stdev, stdevstr, fuzzer->get_int)
			ni.set<uint64_t>(mean, stdev);
		}
		break;
		default:
			// will never happen
		break;
	}
	return out;
}


void iterate (nnet::tensor& ten, std::function<void(size_t,const char*)> iter)
{
	std::string everything;
	TENS_TYPE type = ten.get_type();
	switch (type)
	{
		case DOUBLE:
		{
			STRINGIFY(double)
		}
		break;
		case FLOAT:
		{
			STRINGIFY(float)
		}
		break;
		case INT8:
		{
			STRINGIFY(int8_t)
		}
		break;
		case UINT8:
		{
			STRINGIFY(uint8_t)
		}
		break;
		case INT16:
		{
			STRINGIFY(int16_t)
		}
		break;
		case UINT16:
		{
			STRINGIFY(uint16_t)
		}
		break;
		case INT32:
		{
			STRINGIFY(int32_t)
		}
		break;
		case UINT32:
		{
			STRINGIFY(uint32_t)
		}
		break;
		case INT64:
		{
			STRINGIFY(int64_t)
		}
		break;
		case UINT64:
		{
			STRINGIFY(uint64_t)
		}
		break;
		default:
			// will never happen
		break;
	}
	size_t nbytes = nnet::type_size(type);
	const char* evr = everything.c_str();
	for (size_t i = 0; i < everything.size(); i += nbytes)
	{
		iter(i / nbytes, evr + i);
	}
}


TEST_F(DATA_SRC, Copy_D000)
{
	nnet::const_init ci;
	nnet::r_uniform_init ui;
	nnet::r_normal_init ni;

	nnet::const_init ciassign;
	nnet::r_uniform_init uiassign;
	nnet::r_normal_init niassign;

	// initialize and setup expectations
	std::string cnstr;
	std::string minstr, maxstr;
	std::string meanstr, stdevstr;
	TENS_TYPE ctype = fuzz_const(this, ci, cnstr);
	TENS_TYPE utype = fuzz_uniform(this, ui, minstr, maxstr);
	TENS_TYPE ntype = fuzz_normal(this, ni, meanstr, stdevstr);

	nnet::const_init* cicpy = ci.clone();
	nnet::r_uniform_init* uicpy = ui.clone();
	nnet::r_normal_init* nicpy = ni.clone();

	nnet::GENERIC cval = cicpy->get_const();
	nnet::GENERIC minval = uicpy->get_min();
	nnet::GENERIC maxval = uicpy->get_max();
	nnet::GENERIC meanval = nicpy->get_mean();
	nnet::GENERIC stdevval = nicpy->get_stdev();

	EXPECT_STREQ(cnstr.c_str(), cval.first.c_str());
	EXPECT_EQ(ctype, cval.second);
	EXPECT_STREQ(minstr.c_str(), minval.first.c_str());
	EXPECT_STREQ(maxstr.c_str(), maxval.first.c_str());
	EXPECT_EQ(utype, minval.second);
	EXPECT_EQ(utype, maxval.second);
	EXPECT_STREQ(meanstr.c_str(), meanval.first.c_str());
	EXPECT_STREQ(stdevstr.c_str(), stdevval.first.c_str());
	EXPECT_EQ(ntype, meanval.second);
	EXPECT_EQ(ntype, stdevval.second);

	ciassign = ci;
	uiassign = ui;
	niassign = ni;

	nnet::GENERIC cval2 = ciassign.get_const();
	nnet::GENERIC minval2 = uiassign.get_min();
	nnet::GENERIC maxval2 = uiassign.get_max();
	nnet::GENERIC meanval2 = niassign.get_mean();
	nnet::GENERIC stdevval2 = niassign.get_stdev();

	EXPECT_STREQ(cnstr.c_str(), cval2.first.c_str());
	EXPECT_EQ(ctype, cval2.second);
	EXPECT_STREQ(minstr.c_str(), minval2.first.c_str());
	EXPECT_STREQ(maxstr.c_str(), maxval2.first.c_str());
	EXPECT_EQ(utype, minval2.second);
	EXPECT_EQ(utype, maxval2.second);
	EXPECT_STREQ(meanstr.c_str(), meanval2.first.c_str());
	EXPECT_STREQ(stdevstr.c_str(), stdevval2.first.c_str());
	EXPECT_EQ(ntype, meanval2.second);
	EXPECT_EQ(ntype, stdevval2.second);

	delete cicpy;
	delete uicpy;
	delete nicpy;
}


TEST_F(DATA_SRC, ConstInit_D001)
{
	nnet::tensorshape shape = random_def_shape(this);
	nnet::tensor cten(shape);
	nnet::tensor vten(shape);

	nnet::const_init ci;
	nnet::const_init vi; // vector set
	nnet::const_init badci;

	EXPECT_EQ(BAD_T, ci.get_const().second);
	EXPECT_EQ(BAD_T, vi.get_const().second);

	EXPECT_THROW(badci.set<std::string>("expect failure"), std::exception);
	EXPECT_THROW(badci.set<std::string>(std::vector<std::string>{"expect failure"}), std::exception);

	std::string expect_const;
	std::string expect_vec;
	TENS_TYPE ctype = fuzz_const(this, ci, expect_const);
	TENS_TYPE vtype = fuzz_vec(this, vi, expect_vec, shape.n_elems());

	cten.read_from(ci);
	vten.read_from(vi);

	size_t cnbytes = nnet::type_size(ctype);
	iterate(cten, 
	[&](size_t, const char* c)
	{
		std::string got(c, cnbytes);
		EXPECT_STREQ(expect_const.c_str(), got.c_str());
	});

	size_t vnbytes = nnet::type_size(vtype);
	size_t snippetsize = expect_vec.size() / vnbytes;
	iterate(vten, 
	[&](size_t i, const char* c)
	{
		size_t vidx = i % snippetsize;
		std::string expect(&expect_vec[vidx], vnbytes);
		std::string got(c, vnbytes);
		EXPECT_STREQ(expect.c_str(), got.c_str());
	});
}


TEST_F(DATA_SRC, RandUnif_D002)
{
	nnet::tensorshape shape = random_def_shape(this);
	nnet::tensor ten(shape);

	nnet::r_uniform_init ri;
	nnet::r_uniform_init badi;

	EXPECT_EQ(BAD_T, ri.get_min().second);
	EXPECT_THROW(badi.set<std::string>("min", "max"), std::exception);

	std::string minstr, maxstr;
	TENS_TYPE utype = fuzz_uniform(this, ri, minstr, maxstr);

	ten.read_from(ri);

	iterate(ten, 
	[&](size_t, const char* c)
	{
		switch (utype)
		{
			case DOUBLE:
			{
				double* val = (double*) c;
				double* min = (double*) &minstr[0];
				double* max = (double*) &maxstr[0];
				EXPECT_LE(*min, *val) << "double type";
				EXPECT_GE(*max, *val) << "double type";
			}
			break;
			case FLOAT:
			{
				float* val = (float*) c;
				float* min = (float*) &minstr[0];
				float* max = (float*) &maxstr[0];
				EXPECT_LE(*min, *val) << "float type";
				EXPECT_GE(*max, *val) << "float type";
			}
			break;
			case INT8:
			{
				int8_t* val = (int8_t*) c;
				int8_t* min = (int8_t*) &minstr[0];
				int8_t* max = (int8_t*) &maxstr[0];
				EXPECT_LE(*min, *val) << "int8 type";
				EXPECT_GE(*max, *val) << "int8 type";
			}
			break;
			case UINT8:
			{
				uint8_t* val = (uint8_t*) c;
				uint8_t* min = (uint8_t*) &minstr[0];
				uint8_t* max = (uint8_t*) &maxstr[0];
				EXPECT_LE(*min, *val) << "uint8 type";
				EXPECT_GE(*max, *val) << "uint8 type";
			}
			break;
			case INT16:
			{
				int16_t* val = (int16_t*) c;
				int16_t* min = (int16_t*) &minstr[0];
				int16_t* max = (int16_t*) &maxstr[0];
				EXPECT_LE(*min, *val) << "int16 type";
				EXPECT_GE(*max, *val) << "int16 type";
			}
			break;
			case UINT16:
			{
				uint16_t* val = (uint16_t*) c;
				uint16_t* min = (uint16_t*) &minstr[0];
				uint16_t* max = (uint16_t*) &maxstr[0];
				EXPECT_LE(*min, *val) << "uint16 type";
				EXPECT_GE(*max, *val) << "uint16 type";
			}
			break;
			case INT32:
			{
				int32_t* val = (int32_t*) c;
				int32_t* min = (int32_t*) &minstr[0];
				int32_t* max = (int32_t*) &maxstr[0];
				EXPECT_LE(*min, *val) << "int32 type";
				EXPECT_GE(*max, *val) << "int32 type";
			}
			break;
			case UINT32:
			{
				uint32_t* val = (uint32_t*) c;
				uint32_t* min = (uint32_t*) &minstr[0];
				uint32_t* max = (uint32_t*) &maxstr[0];
				EXPECT_LE(*min, *val) << "uint32 type";
				EXPECT_GE(*max, *val) << "uint32 type";
			}
			break;
			case INT64:
			{
				int64_t* val = (int64_t*) c;
				int64_t* min = (int64_t*) &minstr[0];
				int64_t* max = (int64_t*) &maxstr[0];
				EXPECT_LE(*min, *val) << "int64 type";
				EXPECT_GE(*max, *val) << "int64 type";
			}
			break;
			case UINT64:
			{
				uint64_t* val = (uint64_t*) c;
				uint64_t* min = (uint64_t*) &minstr[0];
				uint64_t* max = (uint64_t*) &maxstr[0];
				EXPECT_LE(*min, *val) << "uint64 type";
				EXPECT_GE(*max, *val) << "uint64 type";
			}
			break;
			default:
			break;
		}
	});
}


// TEST_F(DATA_SRC, RandNorm_D003)
// {

// }


#endif /* DISABLE_DSRC_TEST */


#endif /* DISABLE_TENSOR_MODULE_TESTS */
