//
//  tensor_actor.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2018-01-16.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#ifdef TENNCOR_TENSOR_ACTOR_HPP

namespace nnet
{

// ELEM UNARY

template <typename T>
tens_elem_uni<T>::tens_elem_uni (out_wrapper<void> dest, 
	std::vector<in_wrapper<void> > srcs, UNI_TRANS<T> trans) : 
tens_template<T>(dest, srcs), trans_(trans) {}

template <typename T>
void tens_elem_uni<T>::action (void)
{
	assert(1 == this->srcs_.size());
	size_t n_out = this->dest_.second.n_elems();
	const T* src = this->srcs_[0].first;
	std::transform(src, src + n_out, this->dest_.first, trans_);
}

// ELEM BINARY

template <typename T>
tens_elems_bi<T>::tens_elems_bi (out_wrapper<void> dest, 
	std::vector<in_wrapper<void> > srcs, BI_TRANS<T> trans) : 
tens_template<T>(dest, srcs), trans_(trans) {}

template <typename T>
void tens_elems_bi<T>::action (void)
{
	assert(2 == this->srcs_.size());
	size_t n_out = this->dest_.second.n_elems();
	bool left_mul = this->srcs_[0].second.n_elems() > 1;
	bool right_mul = this->srcs_[1].second.n_elems() > 1;

	for (size_t i = 0; i < n_out; ++i)
	{
		this->dest_.first[i] = trans_(
			this->srcs_[0].first[i * left_mul], 
			this->srcs_[1].first[i * right_mul]);
	}
}

// AXIAL BINARY

template <typename T>
tens_axial_elems<T>::tens_axial_elems (out_wrapper<void> dest, 
	std::vector<in_wrapper<void> > srcs,
	size_t axis, bool left, BI_TRANS<T> trans) :
tens_template<T>(dest, srcs), 
trans_(left ? trans : [trans](T a, T b) { return trans(b, a); }), 
axis_(axis), idx_((short) left) {}

template <typename T>
void tens_axial_elems<T>::action (void)
{
	assert(2 == this->srcs_.size());
	size_t n_elems = this->dest_.second.n_elems();
	T* dest = this->dest_.first;
	std::vector<size_t> olist = this->dest_.second.as_list();
	std::vector<size_t> ilist = this->srcs_[idx_].second.as_list();

	if (1 == this->srcs_[idx_].second.n_elems())
	{
		for (size_t i = 0; i < n_elems; i++)
		{
			dest[i] = trans_(
				this->srcs_[idx_].first[0],
				this->srcs_[(idx_ + 1) % 2].first[i]);
		}
	}
	else
	{
		for (size_t i = 0; i < n_elems; i++)
		{
			size_t axis_idx = i;
			if (axis_ == 0 && ilist[axis_] != olist[axis_])
			{
				std::vector<size_t> coords = this->dest_.second.coordinate_from_idx(i);
				if (ilist[axis_] != olist[axis_ + 1])
				{
					std::stringstream ss;
					ss << "failed to map ";
					print_shape(this->dest_.second, ss);
					ss << " to ";
					print_shape(this->srcs_[idx_].second, ss);
					ss << " along axis 0";
					throw std::logic_error(ss.str());
				}
				coords = std::vector<size_t>(coords.begin()+1, coords.end());
				axis_idx = this->srcs_[idx_].second.flat_idx(coords);
			}
			else if (axis_ >= ilist.size() || (ilist[axis_] == 1 && olist[axis_]> 1))
			{
				std::vector<size_t> coords = this->dest_.second.coordinate_from_idx(i);
				coords[axis_] = 0;
				axis_idx = this->srcs_[idx_].second.flat_idx(coords);
			}
			dest[i] = trans_(this->srcs_[idx_].first[axis_idx], this->srcs_[(idx_ + 1) % 2].first[i]);
		}
	}
}

// CONDITIONAL UNARY

template <typename T>
tens_conditional_uni<T>::tens_conditional_uni (out_wrapper<void> dest, 
	std::vector<in_wrapper<void> > srcs, UNI_COMP<T> comp) : 
tens_template<T>(dest, srcs), comp_(comp) {}

template <typename T>
void tens_conditional_uni<T>::action (void)
{
	assert(1 == this->srcs_.size());
	size_t n_out = this->dest_.second.n_elems();
	const T* src = this->srcs_[0].first;
	std::transform(src, src + n_out, this->dest_.first, comp_);
}

// CONDITIONAL BINARY

template <typename T>
tens_conditional<T>::tens_conditional (out_wrapper<void> dest, 
	std::vector<in_wrapper<void> > srcs, COMPARE<T> comp) : 
tens_template<T>(dest, srcs), comp_(comp) {}

template <typename T>
void tens_conditional<T>::action (void)
{
	assert(2 == this->srcs_.size());
	size_t n_out = this->dest_.second.n_elems();
	// assert everything shape is ok
	bool left_mul = this->srcs_[0].second.n_elems() > 1;
	bool right_mul = this->srcs_[1].second.n_elems() > 1;

	for (size_t i = 0; i < n_out; ++i)
	{
		this->dest_.first[i] = comp_(
			this->srcs_[0].first[i * left_mul], 
			this->srcs_[1].first[i * right_mul]);
	}
}

// GENERAL

template <typename T>
tens_general<T>::tens_general (out_wrapper<void> dest, 
	std::vector<in_wrapper<void> > srcs, GEN_TRANS<T> trans) : 
tens_template<T>(dest, srcs), trans_(trans) {}

template <typename T>
void tens_general<T>::action (void)
{
	trans_(this->dest_, this->srcs_);
}

}

#endif
