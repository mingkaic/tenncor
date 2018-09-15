#include "ade/test/common.hpp"
#include <iostream>
std::vector<ade::DimT> get_shape (SESSION& sess, std::string label)
{
	int32_t n = sess->get_scalar("n_" + label, {1, ade::rank_cap - 1});
	int32_t max_elem = std::log(nelem_limit) / std::log(n);
	max_elem = std::max(3, max_elem);
	auto temp = sess->get_int(label, n, {2, max_elem});
	return std::vector<ade::DimT>(temp.begin(), temp.end());
}

std::vector<ade::DimT> get_zeroshape (SESSION& sess, std::string label)
{
	int32_t nz = sess->get_scalar("n_" + label, {1, ade::rank_cap - 1});
	int32_t max_zelem = std::log(nelem_limit) / std::log(nz);
	max_zelem = std::max(3, max_zelem);
	auto temp = sess->get_int(label, nz, {0, max_zelem});
	int32_t zidx = 0;
	if (nz > 1)
	{
		zidx = sess->get_scalar(label + "_idx", {0, nz - 1});
	}
	temp[zidx] = 0;
	return std::vector<ade::DimT>(temp.begin(), temp.end());
}

std::vector<ade::DimT> get_longshape (SESSION& sess, std::string label)
{
	int32_t nl = sess->get_scalar("n_" + label, {ade::rank_cap, 57});
	int32_t max_lelem = std::log(nelem_limit) / std::log(nl);
	max_lelem = std::max(3, max_lelem);
	auto temp = sess->get_int(label, nl, {1, max_lelem});
	return std::vector<ade::DimT>(temp.begin(), temp.end());
}
