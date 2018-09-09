#include "util/strify.hpp"

#include "ade/shape.hpp"

// #include "simple/jack.hpp"

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

// std::vector<ade::DimT> get_shape (SESSION& sess, std::string label)
// {
// 	long n = sess->get_scalar("n_" + label, {0, ade::rank_cap});
// 	size_t max_elem = std::log(nelem_limit) / std::log(n);
// 	auto temp = sess->get_int(label, n, {1, max_elem});
// 	return std::vector<ade::DimT>(temp.begin(), temp.end());
// }

// std::vector<ade::DimT> get_zeroshape (SESSION& sess, std::string label)
// {
// 	long nz = sess->get_scalar("n_" + label, {0, ade::rank_cap});
// 	size_t max_zelem = std::log(nelem_limit) / std::log(nz);
// 	auto temp = sess->get_int(label, n, {0, max_zelem});
// 	long zidx = sess->get_scalar(label + "_idx", {0, nz});
// 	temp[zidx] = 0;
// 	return std::vector<ade::DimT>(temp.begin(), temp.end());
// }

// std::vector<ade::DimT> get_longshape (SESSION& sess, std::string label)
// {
// 	long nl = sess->get_scalar("n_" + label, {ade::rank_cap, 57});
// 	size_t max_lelem = std::log(nelem_limit) / std::log(nl);
// 	auto temp = sess->get_int(label, n, {1, max_lelem});
// 	return std::vector<ade::DimT>(temp.begin(), temp.end());
// }
