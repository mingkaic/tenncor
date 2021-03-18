#include "internal/utils/sparse/random.hpp"

#ifdef EIGEN_SPARSE_RANDOM_HPP

namespace eigen
{

static const size_t chunklimit = 10000;

static void random_indices_helper (
	std::vector<size_t>::iterator obegin,
	std::vector<size_t>::iterator oend,
	size_t begin, size_t end,
	const global::GenPtrT& generator)
{
	if (obegin == oend || end == begin)
	{
		return;
	}
	size_t n = end - begin;
	size_t on = std::distance(obegin, oend);
	assert(n >= on);
	if (on == 1)
	{
		*obegin = generator->unif_int(begin, end);
		return;
	}
	std::vector<size_t> indices(n);
	auto it = indices.begin();
	std::iota(it, indices.end(), begin);
	fisher_yates_shuffle(it, indices.end(), on, generator);
	std::copy(it, it + on, obegin);
}

void random_indices (
	std::vector<size_t>::iterator begin,
	std::vector<size_t>::iterator end, size_t n,
	const global::GenPtrT& generator)
{
	size_t nouts = std::distance(begin, end);
	assert(n > nouts);

	size_t nchunks = n / chunklimit;
	if (nouts < nchunks)
	{
		// there are more chunks than output size, so select chunks at random
		std::vector<size_t> chunk_indices(nouts);
		random_indices(chunk_indices.begin(), chunk_indices.end(),
			nchunks, generator);
		size_t i = 0;
		for (auto it = begin; it != end; ++it)
		{
			size_t chunk = chunk_indices[i];
			auto cbegin = chunk * chunklimit;
			auto cend = std::min(cbegin + chunklimit, n);
			random_indices_helper(it, it + 1, cbegin, cend, generator);
			++i;
		}
		return;
	}

	if (nchunks > 0)
	{
		size_t outchunksize = nouts / nchunks;
		for (size_t chunk = 0; chunk < nchunks; ++chunk)
		{
			auto cbegin = chunk * chunklimit;
			auto cend = std::min(cbegin + chunklimit, n);
			auto next = begin + outchunksize;
			random_indices_helper(begin, next, cbegin, cend, generator);
			begin = next;
		}
	}
	// randomize remainding data
	random_indices_helper(begin, end, nchunks * chunklimit, n, generator);
}

}

#endif
