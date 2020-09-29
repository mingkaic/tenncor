
#ifndef DISTR_OX_SEGMENT_HPP
#define DISTR_OX_SEGMENT_HPP

#include "tenncor/distr/consul.hpp"

#include "tenncor/serial/oxsvc/topography.hpp"

namespace segment
{

using KSelectF = std::function<std::vector<size_t>(size_t,
	const types::StrUMapT<size_t>&)>;

static const size_t max_try_kmeans = 10;

template <typename T>
std::function<T&(size_t,size_t)> sqr_curry (
	std::vector<T>& src, size_t n)
{
	return [&src, n](size_t i, size_t j) -> T&
	{
		return src[i + j * n];
	};
}

void floyd_warshall (std::vector<size_t>& distance,
	types::StrUMapT<size_t>& vertices,
	const distr::ox::GraphT& nodes);

std::vector<distr::ox::GraphT> disjoint_graphs (const distr::ox::GraphT& graph);

distr::ox::TopographyT kmeans (
	const types::StringsT& peers,
	const distr::ox::GraphT& nodes,
	KSelectF select =
	[](size_t k, const types::StrUMapT<size_t>& vertices)
	{
		size_t n = vertices.size();

		// randomly select k initial means
		std::random_device rd;
		std::mt19937 gen(rd());
		std::vector<size_t> cands(n);
		std::iota(cands.begin(), cands.end(), 0);
		std::shuffle(cands.begin(), cands.end(), gen);

		return std::vector<size_t>(cands.begin(), cands.begin() + k);
	});

}

#endif // DISTR_OX_SEGMENT_HPP
