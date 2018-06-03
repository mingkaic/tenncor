/*!
 *
 *  placeholder.hpp
 *  wire
 *
 *  Purpose:
 *  variable wrapper with vector assignment
 *
 *  Created by Mingkai Chen on 2016-11-08
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "ioutil/stream.hpp"

#include "mold/variable.hpp"

#include "slip/error.hpp"

#include "wire/identifier.hpp"

#pragma once
#ifndef WIRE_PLACEHOLDER_HPP
#define WIRE_PLACEHOLDER_HPP

namespace wire
{

class Placeholder final : public Identifier
{
public:
	Placeholder (std::string label,
		Graph& graph = Graph::get_global());

	Placeholder (clay::Shape shape, std::string label,
		Graph& graph = Graph::get_global());

	Placeholder& operator = (const Placeholder&) = default;

	Placeholder& operator = (Placeholder&& other) = default;

	//! assign raw data according to a
	//! vector representation of inner tensor
	//! for a shape of <d_0, d_1, ..., d_i> and
	//! 	coordinate <c_0, c_1, ..., c_i>:
	//! index mapping function is
	//! sum_j=0:i(product_k=0:j(d_k-1) * c_j) where for k < 0 d_k = 1
	template <typename T>
	Placeholder& operator = (std::vector<T> data)
	{
		size_t n = data.size();
		mold::Variable* arg = static_cast<mold::Variable*>(get());
		if (false == arg->has_data())
		{
			clay::DTYPE dtype = clay::get_type<T>();
			std::shared_ptr<clay::iBuilder> bptr;
			auto it = graph_->alloweds_.find(get_uid());
			clay::Shape allowed;
			if (graph_->alloweds_.end() != it)
			{
				allowed = it->second;
				graph_->alloweds_.erase(it);
			}
			RawBuilder builder(n, dtype);
			if (allowed.is_fully_defined())
			{
				arg->initialize(builder, allowed);
			}
			else
			{
				arg->initialize(builder);
			}
		}
		clay::State state = arg->get_state();
		assert(state.shape_.is_fully_defined());
		if (n > state.shape_.n_elems())
		{
			throw std::logic_error(ioutil::Stream() << "data with "
				<< n << " elements cannot be assigned to allcoated tensor with "
				<< state.shape_.n_elems() << " elements");
		}
		clay::DTYPE dtype = clay::get_type<T>();
		if (dtype != state.dtype_)
		{
			throw slip::TypeMismatchError(state.dtype_, dtype);
		}
		std::string s((char*) &data[0], n * sizeof(T));
		AssignIO assign(s, state.shape_, state.dtype_);
		arg->assign(assign);
		return *this;
	}

	Identifier* derive (Identifier* wrt) override;

private:
	struct AssignIO final : public clay::iSource
	{
		AssignIO (std::string data,
			clay::Shape shape, clay::DTYPE dtype);

		bool read_data (clay::State& dest) const override;

	private:
		std::string data_;

		clay::Shape shape_;

		clay::DTYPE dtype_;
	};

	struct RawBuilder final : public clay::iBuilder
	{
		RawBuilder (size_t limit, clay::DTYPE dtype) :
			limit_(limit), dtype_(dtype) {}

		clay::TensorPtrT get (void) const override;

		clay::TensorPtrT get (clay::Shape shape) const override;

	protected:
		clay::iBuilder* clone_impl (void) const override;

	private:
		size_t limit_;

		clay::DTYPE dtype_;
	};
};

}

#endif /* WIRE_PLACEHOLDER_HPP */
