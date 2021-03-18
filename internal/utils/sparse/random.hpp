#include "internal/utils/coord/coord.hpp"

#include "internal/eigen/eigen.hpp"

#ifndef EIGEN_SPARSE_RANDOM_HPP
#define EIGEN_SPARSE_RANDOM_HPP

namespace eigen
{

template <typename IT>
using ItValT = typename std::iterator_traits<IT>::value_type;

// source: https://ideone.com/3A3cv
template <typename IT>
IT fisher_yates_shuffle (IT begin, IT end, size_t m,
	const global::GenPtrT& generator = global::get_generator())
{
	size_t left = std::distance(begin, end);
	assert(m < left);
	auto gen = generator->unif_intgen(0, left - 1);
	while (m--)
	{
		IT r = begin;
		size_t adv = gen() % left;
		std::advance(r, adv);
		std::swap(*begin, *r);
		++begin;
		--left;
	}
	return begin;
}

// populate [obegin, oend] with indices up to n
void random_indices (std::vector<size_t>::iterator obegin,
	std::vector<size_t>::iterator oend, size_t n,
	const global::GenPtrT& generator = global::get_generator());

template <typename T>
void identity_sparse (eigen::TripletsT<T>& out,
	const teq::Shape& shape, global::GenF<T> gen)
{
	size_t n = std::min(shape.at(0), shape.at(1));
	out.reserve(n);
	for (size_t i = 0; i < n; ++i)
	{
		out.push_back(eigen::TripletT<T>(i, i, gen()));
	}
}

template <typename T,
	std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
void random_sparse (eigen::TripletsT<T>& out,
	const teq::Shape& shape, double density,
	const global::GenPtrT& generator = global::get_generator())
{
	size_t nzs = shape.n_elems() * density;
	auto gen = generator->unif_decgen(-1, 1);

	std::vector<size_t> indices(nzs);
	random_indices(indices.begin(), indices.end(), shape.n_elems(), generator);
	out.reserve(nzs);
	for (size_t i : indices)
	{
		auto coord = teq::coordinate(shape, i);
		out.push_back(eigen::TripletT<float>(
			coord[1], coord[0], gen()));
	}
}

}

#endif // EIGEN_SPARSE_RANDOM_HPP
