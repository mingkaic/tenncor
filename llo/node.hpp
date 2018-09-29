/*!
 *
 *  node.hpp
 *  llo
 *
 *  Purpose:
 *  define extensions to ade::iTensor that evaluate data
 *
 */

#include <cassert>

#include "ade/functor.hpp"

#include "llo/opmap.hpp"

#ifndef LLO_NODE_HPP
#define LLO_NODE_HPP

namespace llo
{

/*! Evaluable interface for calculating data of subtree */
struct iEvaluable
{
	virtual ~iEvaluable (void) = default;

	/*! calculate data of operation subtree and return in the specified type */
	virtual GenericData evaluate (DTYPE dtype) = 0;

	/*! calculate data of operation subtree and return in the specified type */
	virtual ade::Tensorptr inner (void) const = 0;
};

/*! Evaluate Tensors and get return data of input type */
GenericData evaluate (DTYPE dtype, ade::iTensor* tens);

/*! Tensor evaluable interface for representing leaf nodes with data */
struct iSource : public ade::iTensor, public iEvaluable
{
	virtual ~iSource (void) = default;

	/*! get type of data held by the source */
	virtual DTYPE native_type (void) const = 0;

	/*! assign data to the source */
	virtual void reassign (const GenericRef& data) = 0;
};

/*! Source implementation holding data */
template <typename T>
struct Source final : public iSource
{
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

	/*! implementation of iTensor */
	const ade::Shape& shape (void) const override
	{
		return tens_->shape();
	}

	/*! implementation of iTensor */
	ade::Tensorptr gradient (ade::Tensorptr& wrt) const override
	{
		iEvaluable* eval = dynamic_cast<iEvaluable*>(wrt.get());
		ade::Tensorptr target = nullptr == eval ? wrt : eval->inner();
		return tens_->gradient(target);
	}

	/*! implementation of iTensor */
	std::string to_string (void) const override
	{
		return tens_->to_string();
	}

	/*! implementation of iEvaluable */
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

	/*! implementation of iEvaluable */
	ade::Tensorptr inner (void) const override
	{
		return tens_;
	}

	/*! implementation of iSource */
	DTYPE native_type (void) const override
	{
		return get_type<T>();
	}

	/*! implementation of iSource */
	void reassign (const GenericRef& data) override
	{
		assert(data.shape_.compatible_after(tens_->shape(), 0));
		std::memcpy(&data_[0], data.data_, sizeof(T) * data.shape_.n_elems());
	}

private:
	Source (ade::Shape shape, std::vector<T>& data) :
		tens_(ade::Tensor::get(shape)), data_(data) {}

	/*! tensor holding shape info */
	ade::Tensorptr tens_;
	/*! data vector */
	std::vector<T> data_;
};

/*! Tensorptr extension to provide data assignment functionality */
template <typename T>
struct Placeholder final : public ade::Tensorptr
{
	Placeholder (ade::Shape shape) : ade::Tensorptr(
		Source<T>::get(shape, std::vector<T>(shape.n_elems()))) {}

	Placeholder (const Placeholder&) = default;
	Placeholder (Placeholder&&) = default;
	Placeholder& operator = (const Placeholder&) = default;
	Placeholder& operator = (Placeholder&&) = default;

	/*! assign data to interally referenced source */
	Placeholder& operator = (std::vector<T>& data)
	{
		auto src = static_cast<iSource*>(ptr_.get());
		GenericRef gdata((char*) &data[0], src->shape(), get_type<T>());
		src->reassign(gdata);
		return *this;
	}
};

/*! Functor evaluable implementation for executing data operations */
template <typename... Args>
struct DirectWrapper final : public ade::iFunctor, public iEvaluable
{
	static ade::Tensorptr get (ade::Tensorptr tens, Args... args)
	{
		if (nullptr == dynamic_cast<ade::iFunctor*>(tens.get()))
		{
			util::handle_error("wrapping non-functor");
		}
		std::tuple<Args...> tp(args...);
		return new DirectWrapper(tens, tp);
	}

	/*! implementation of iTensor  */
	const ade::Shape& shape (void) const override
	{
		return tens_->shape();
	}

	/*! implementation of iTensor  */
	ade::Tensorptr gradient (ade::Tensorptr& wrt) const override
	{
		if (iEvaluable* eval = dynamic_cast<iEvaluable*>(wrt.get()))
		{
			ade::Tensorptr wrt = eval->inner();
			return tens_->gradient(wrt);
		}
		return tens_->gradient(wrt);
	}

	/*! implementation of iTensor  */
	std::string to_string (void) const override
	{
		return "Wrapper_" + opname(get_code()) + "<" + util::tuple_to_string(args_) + ">";
	}

	/*! implementation of iFunctor  */
	ade::OPCODE get_code (void) const override
	{
		return static_cast<ade::iFunctor*>(tens_.get())->get_code();
	}

	/*! implementation of iFunctor  */
	std::vector<ade::iTensor*> get_refs (void) const override
	{
		return static_cast<ade::iFunctor*>(tens_.get())->get_refs();
	}

	/*! implementation of iEvaluable  */
	GenericData evaluate (DTYPE dtype) override
	{
		return eval_helper(dtype, std::index_sequence_for<Args...>());
	}

	/*! implementation of iEvaluable  */
	ade::Tensorptr inner (void) const override
	{
		return tens_;
	}

	/*! non-tensor metadata used by certain data operations  */
	std::tuple<Args...> args_;

private:
	DirectWrapper (ade::Tensorptr& tens, std::tuple<Args...>& args) :
		args_(args), tens_(tens) {}

	template <size_t... I>
	GenericData eval_helper (DTYPE dtype, std::index_sequence<I...>) const
	{
		ade::iFunctor* f = static_cast<ade::iFunctor*>(tens_.get());
		ade::OPCODE opcode = f->get_code();

		std::vector<ade::iTensor*> refs = f->get_refs();
		uint8_t nargs = refs.size();

		GenericData out(f->shape(), dtype);
		std::vector<GenericData> argdata(nargs);
		for (uint8_t i = 0; i < nargs; ++i)
		{
			argdata[i] = llo::evaluate(dtype, refs[i]);
		}
		op_exec(opcode, out, argdata, std::get<I>(args_)...);
		return out;
	}

	/*! tensor proxy source */
	ade::Tensorptr tens_;
};

template <typename T>
ade::Tensorptr scalar (T value)
{
	return llo::Source<T>::get(ade::Shape(), {value});
}

}

#endif /* LLO_NODE_HPP */
