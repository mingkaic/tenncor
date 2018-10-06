///
/// node.hpp
/// llo
///
/// Purpose:
/// Extend ade::iTensor with proxies that carry and evaluate tensor data
///

#include <cassert>

#include "ade/functor.hpp"

#include "llo/opmap.hpp"

#ifndef LLO_NODE_HPP
#define LLO_NODE_HPP

namespace llo
{

/// Interface for ensuring data evaluation functionality for proxies
struct iEvaluable
{
	virtual ~iEvaluable (void) = default;

	/// Return the result of the equation subgraph of input type
	virtual GenericData evaluate (DTYPE dtype) = 0;

	// Return the internal ade::Tensorptr
	virtual ade::Tensorptr inner (void) const = 0;
};

/// Return the evaluate result of ade::Tensorptr polymorphically
GenericData evaluate (DTYPE dtype, ade::iTensor* tens);

/// Extended interface for leaf proxies with tensor data
struct iSource : public ade::iTensor, public iEvaluable
{
	virtual ~iSource (void) = default;

	/// Return the type of data stored
	virtual DTYPE native_type (void) const = 0;

	/// Assign new data values
	virtual void reassign (const GenericRef& data) = 0;
};

/// Leaf proxy holding tensor data
template <typename T>
struct Source final : public iSource
{
	/// Return a Source of input shape and containing input data
	static ade::Tensorptr get (ade::Shape shape, std::vector<T> data)
	{
		if (shape.n_elems() != data.size())
		{
			util::handle_error("data size does not match shape",
				util::ErrArg<size_t>("data.size", data.size()),
				util::ErrArg<size_t>("shape.n_elems", shape.n_elems()));
		}
		return new Source(shape, data);
	}

	/// Implementation of iTensor
	const ade::Shape& shape (void) const override
	{
		return tens_->shape();
	}

	/// Implementation of iTensor
	ade::Tensorptr gradient (ade::Tensorptr& wrt) const override
	{
		iEvaluable* eval = dynamic_cast<iEvaluable*>(wrt.get());
		ade::Tensorptr target = nullptr == eval ? wrt : eval->inner();
		return tens_->gradient(target);
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return tens_->to_string();
	}

	/// Implementation of iEvaluable
	GenericData evaluate (DTYPE dtype) override
	{
		DTYPE curtype = get_type<T>();
		GenericData out(tens_->shape(), curtype);
		std::memcpy(out.data_.get(), &data_[0], sizeof(T) * data_.size());
		if (curtype != dtype)
		{
			return out.convert_to(dtype);
		}
		return out;
	}

	/// Implementation of iEvaluable
	ade::Tensorptr inner (void) const override
	{
		return tens_;
	}

	/// Implementation of iSource
	DTYPE native_type (void) const override
	{
		return get_type<T>();
	}

	/// Implementation of iSource
	void reassign (const GenericRef& data) override
	{
		assert(data.shape_.compatible_after(tens_->shape(), 0));
		std::memcpy(&data_[0], data.data_, sizeof(T) * data.shape_.n_elems());
	}

private:
	Source (ade::Shape shape, std::vector<T>& data) :
		tens_(ade::Tensor::get(shape)), data_(data) {}

	/// Tensor holding shape info
	ade::Tensorptr tens_;

	/// Tensor data
	std::vector<T> data_;
};

/// Source smartpointer to provide data assignment functionality as Tensorptr
template <typename T>
struct Placeholder final : public ade::Tensorptr
{
	Placeholder (ade::Shape shape) : ade::Tensorptr(
		Source<T>::get(shape, std::vector<T>(shape.n_elems()))) {}

	Placeholder (const Placeholder&) = default;
	Placeholder (Placeholder&&) = default;
	Placeholder& operator = (const Placeholder&) = default;
	Placeholder& operator = (Placeholder&&) = default;

	/// Assign vectorized data to source
	Placeholder& operator = (std::vector<T>& data)
	{
		auto src = static_cast<iSource*>(ptr_.get());
		GenericRef gdata((char*) &data[0], src->shape(), get_type<T>());
		src->reassign(gdata);
		return *this;
	}
};

/// Extend ade::Functor for operators requiring
/// non-tensor meta data not needed in ade
/// For example, ade::FLIP has the same output shape as input shape,
/// but the dimension value is needed when evaluating data
template <typename... ARGS>
struct DirectWrapper final : public ade::iFunctor, public iEvaluable
{
	/// Return direct wrapper proxying input tensor and using input metadata
	static ade::Tensorptr get (ade::Tensorptr tens, ARGS... args)
	{
		if (nullptr == dynamic_cast<ade::iFunctor*>(tens.get()))
		{
			util::handle_error("wrapping non-functor");
		}
		std::tuple<ARGS...> tp(args...);
		return new DirectWrapper(tens, tp);
	}

	/// Implementation of iTensor
	const ade::Shape& shape (void) const override
	{
		return tens_->shape();
	}

	/// Implementation of iTensor
	ade::Tensorptr gradient (ade::Tensorptr& wrt) const override
	{
		if (iEvaluable* eval = dynamic_cast<iEvaluable*>(wrt.get()))
		{
			ade::Tensorptr wrt = eval->inner();
			return tens_->gradient(wrt);
		}
		return tens_->gradient(wrt);
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return "Wrapper_" + opname(get_code()) + "<" + util::tuple_to_string(meta_) + ">";
	}

	/// Implementation of iFunctor
	ade::OPCODE get_code (void) const override
	{
		return static_cast<ade::iFunctor*>(tens_.get())->get_code();
	}

	/// Implementation of iFunctor
	std::vector<ade::iTensor*> get_children (void) const override
	{
		return static_cast<ade::iFunctor*>(tens_.get())->get_children();
	}

	/// Implementation of iEvaluable
	GenericData evaluate (DTYPE dtype) override
	{
		return eval_helper(dtype, std::index_sequence_for<ARGS...>());
	}

	/// Implementation of iEvaluable
	ade::Tensorptr inner (void) const override
	{
		return tens_;
	}

	/// Return extra non-tensor arguments
	const std::tuple<ARGS...>& meta (void) const
	{
		return meta_;
	}

private:
	DirectWrapper (ade::Tensorptr& tens, std::tuple<ARGS...>& args) :
		tens_(tens), meta_(args) {}

	template <size_t... I>
	GenericData eval_helper (DTYPE dtype, std::index_sequence<I...>) const
	{
		ade::iFunctor* f = static_cast<ade::iFunctor*>(tens_.get());
		ade::OPCODE opcode = f->get_code();

		std::vector<ade::iTensor*> refs = f->get_children();
		uint8_t nargs = refs.size();

		GenericData out(f->shape(), dtype);
		std::vector<GenericData> argdata(nargs);
		for (uint8_t i = 0; i < nargs; ++i)
		{
			argdata[i] = llo::evaluate(dtype, refs[i]);
		}
		op_exec(opcode, out, argdata, std::get<I>(meta_)...);
		return out;
	}

	/// Tensor proxy source
	ade::Tensorptr tens_;

	/// Extra arguments for certain operators
	/// These arguments are hidden to ensure shape is correct
	/// since meta data can influence shape
	std::tuple<ARGS...> meta_;
};

/// Return a source instance with a scalar as tensor data
template <typename T>
ade::Tensorptr scalar (T value)
{
	return llo::Source<T>::get(ade::Shape(), {value});
}

}

#endif // LLO_NODE_HPP
