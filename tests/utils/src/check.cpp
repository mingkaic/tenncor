#include "check.hpp"

#ifdef TTEST_CHECK_HPP

namespace testutils
{

bool tshape_equal (
	const nnet::tshape& ts1,
	const nnet::tshape& ts2)
{
	std::vector<size_t> vs = ts1.as_list();
	std::vector<size_t> vs2 = ts2.as_list();
	if (vs.size() != vs2.size()) return false;
	return std::equal(vs.begin(), vs.end(), vs2.begin());
}

bool tshape_equal (
	const nnet::tshape& ts1,
	std::vector<size_t>& ts2)
{
	std::vector<size_t> vs = ts1.as_list();
	if (vs.size() != ts2.size()) return false;
	return std::equal(vs.begin(), vs.end(), ts2.begin());
}

}

#endif
