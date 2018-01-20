//
//  tens_elems_bi.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2018-01-19.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#ifdef TENNCOR_TENS_ELEMS_BI_HPP

namespace nnet
{

template <typename T>
tens_bin_sample<T>::tens_bin_sample (out_wrapper<void> dest, 
	std::vector<in_wrapper<void> > srcs) :
tens_elems_bi<T>(dest, srcs, [](T n, T p)
{
	assert(p>= 0 && p <= 1);
	std::binomial_distribution<int> dist(n, p);
	return dist(nnutils::get_generator());
}) {}

template <typename T>
tens_add<T>::tens_add (out_wrapper<void> dest, 
	std::vector<in_wrapper<void> > srcs) :
tens_elems_bi<T>(dest, srcs, [](T a, T b)
{ return a + b; }) {}

template <typename T>
tens_sub<T>::tens_sub (out_wrapper<void> dest, 
	std::vector<in_wrapper<void> > srcs) :
tens_elems_bi<T>(dest, srcs, [](T a, T b)
{ return a - b; }) {}

template <typename T>
tens_mul<T>::tens_mul (out_wrapper<void> dest, 
	std::vector<in_wrapper<void> > srcs) :
tens_elems_bi<T>(dest, srcs, 
[](T a, T b) { return a * b; }) {}

template <typename T>
tens_div<T>::tens_div (out_wrapper<void> dest, 
	std::vector<in_wrapper<void> > srcs) :
tens_elems_bi<T>(dest, srcs, 
[](T a, T b) { return a / b; }) {}

template <typename T>
tens_axial_add<T>::tens_axial_add (out_wrapper<void> dest, 
	std::vector<in_wrapper<void> > srcs, 
	size_t axis) :
tens_axial_elems<T>(dest, srcs, axis, true, 
[](T a, T b) { return a + b; }) {}

template <typename T>
tens_axial_sub<T>::tens_axial_sub (out_wrapper<void> dest, 
	std::vector<in_wrapper<void> > srcs, 
	size_t axis, bool left) :
tens_axial_elems<T>(dest, srcs, axis, left, 
[](T a, T b) { return a - b; }) {}

template <typename T>
tens_axial_mul<T>::tens_axial_mul (out_wrapper<void> dest, 
	std::vector<in_wrapper<void> > srcs, 
	size_t axis) :
tens_axial_elems<T>(dest, srcs, axis, true, 
[](T a, T b) { return a * b; }) {}

template <typename T>
tens_axial_div<T>::tens_axial_div (out_wrapper<void> dest, 
	std::vector<in_wrapper<void> > srcs, 
	size_t axis, bool left) :
tens_axial_elems<T>(dest, srcs, axis, left, 
[](T a, T b) { return a / b; }) {}

}

#endif
