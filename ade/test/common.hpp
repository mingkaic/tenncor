#include "util/strify.hpp"

#include "ade/shape.hpp"

#include "simple/jack.hpp"

const size_t nelem_limit = 32456;

template <typename VEC>
void EXPECT_ARREQ (VEC arr, VEC arr2)
{
	std::stringstream arrs, arrs2;
	util::to_stream(arrs, arr);
	util::to_stream(arrs2, arr2);
	EXPECT_TRUE(std::equal(arr.begin(), arr.end(), arr2.begin())) <<
		"expect list " << arrs.str() << ", got " << arrs2.str() << " instead";
}

std::vector<ade::DimT> get_shape (SESSION& sess, std::string label);

std::vector<ade::DimT> get_zeroshape (SESSION& sess, std::string label);

std::vector<ade::DimT> get_longshape (SESSION& sess, std::string label);
