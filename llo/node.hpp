///
/// node.hpp
/// llo
///
/// Purpose:
/// Extend ade::iTensor with proxies that carry and evaluate tensor data
///

#include "llo/eval.hpp"

#ifndef LLO_NODE_HPP
#define LLO_NODE_HPP

namespace llo
{

/// Leaf evaluable holding tensor data
template <typename T>
struct Source final : public iSource
{
	/// Return a Source of input shape and containing input data
	static DataNode get (ade::Shape shape, std::vector<T> data)
	{
		if (shape.n_elems() != data.size())
		{
			ade::fatalf("data size %d does not match shape %s",
				data.size(), shape.to_string().c_str());
		}
		auto tens = std::shared_ptr<ade::Tensor>(ade::Tensor::get(shape));
		auto src = std::shared_ptr<iSource>(new Source(tens, data));
		return DataNode(EvalCtx(tens.get(), src),
			std::static_pointer_cast<ade::iTensor>(tens));
	}

	/// Return a source instance with a scalar as tensor data
	static DataNode get_scalar (T value)
	{
		return Source<T>::get(ade::Shape(), {value});
	}

	static DataNode copy (const llo::Source<T>& other)
	{
		auto rawsrc = new Source<T>(other);
		auto src = std::shared_ptr<iSource>(rawsrc);
		std::shared_ptr<ade::Tensor> tens = rawsrc->inner();
		return DataNode(EvalCtx(tens.get(), src),
			std::static_pointer_cast<ade::iTensor>(tens));
	}

	/// Implementation of iSource
	GenericData data (DTYPE dtype) const override
	{
		DTYPE curtype = get_type<T>();
		GenericData out(tensor_->shape(), curtype);
		std::memcpy(out.data_.get(), &data_[0], sizeof(T) * data_.size());
		if (curtype != dtype)
		{
			return out.convert_to(dtype);
		}
		return out;
	}

	/// Implementation of iSource
	DTYPE native_type (void) const override
	{
		return get_type<T>();
	}

	/// Implementation of iSource
	void reassign (const GenericRef& data) override
	{
		const ade::Shape& internal_shape = tensor_->shape();
		if (false == data.shape_.compatible_after(internal_shape, 0))
		{
			ade::fatalf("cannot assign data of incompatible shaped %s to "
				"internal data of shape %s", data.shape_.to_string().c_str(),
				internal_shape.to_string().c_str());
		}
		std::memcpy(&data_[0], data.data_, sizeof(T) * data.shape_.n_elems());
	}

	/// Return internal tensor referencing this
	const std::shared_ptr<ade::Tensor>& inner (void) const
	{
		return tensor_;
	}

private:
	Source (std::shared_ptr<ade::Tensor>& tensor, std::vector<T>& data) :
		tensor_(tensor), data_(data) {}

	Source (const Source<T>& other) :
		tensor_(other.tensor_), data_(other.data_) {}

	/// Tensor node of subgraph
	std::shared_ptr<ade::Tensor> tensor_;

	/// Tensor data
	std::vector<T> data_;
};

/// DataNode of a leaf that can be assignable by data vectors
template <typename T>
struct PlaceHolder : public DataNode
{
	PlaceHolder (ade::Shape shape) :
		DataNode(Source<T>::get(shape, std::vector<T>(shape.n_elems()))) {}

	PlaceHolder (const PlaceHolder&) = default;
	PlaceHolder (PlaceHolder&&) = default;
	PlaceHolder& operator = (const PlaceHolder&) = default;
	PlaceHolder& operator = (PlaceHolder&&) = default;

	/// Assign vectorized data to source
	PlaceHolder& operator = (std::vector<T>& data)
	{
		auto key = static_cast<ade::Tensor*>(tensor_.get());
		auto src = ctx_.srcs_[key];
		GenericRef gdata((char*) &data[0], tensor_->shape(), get_type<T>());
		src->reassign(gdata);
		return *this;
	}
};

/// Wrap ade::Functor for operators with non-shape related meta data
/// For example, ade::FLIP has the same output shape as input shape,
/// but the dimension value is needed when evaluating data
template <typename... ARGS>
struct FuncWrapper final : public iEvaluable
{
	/// Return direct function wrapper of input functor and metadata
	static DataNode get (EvalCtx ctx,
		std::shared_ptr<ade::iFunctor> func, ARGS... args)
	{
		std::tuple<ARGS...> tp(args...);
		auto wrapper = std::shared_ptr<iEvaluable>(new FuncWrapper(ctx, func, tp));
		ctx.funks_[func.get()] = wrapper;
		return DataNode(ctx, std::static_pointer_cast<ade::iTensor>(func));
	}

	/// Implementation of iEvaluable
	GenericData data (DTYPE dtype) const override
	{
		return eval_helper(dtype, std::index_sequence_for<ARGS...>());
	}

	/// Return extra non-tensor arguments
	const std::tuple<ARGS...>& meta (void) const
	{
		return meta_;
	}

private:
	FuncWrapper (const EvalCtx& ctx,
		std::shared_ptr<ade::iFunctor> func, std::tuple<ARGS...>& args) :
		ctx_(ctx), func_(func), meta_(args) {}

	template <size_t... I>
	GenericData eval_helper (DTYPE dtype, std::index_sequence<I...>) const
	{
		ade::OPCODE opcode = func_->get_code();
		GenericData out(func_->shape(), dtype);
		std::vector<GenericData> argdata;
		get_func_children(argdata, ctx_, dtype, func_.get());
		op_exec(opcode, out, argdata, std::get<I>(meta_)...);
		return out;
	}

	EvalCtx ctx_;

	/// Tensor proxy source
	std::shared_ptr<ade::iFunctor> func_;

	/// Extra arguments for certain operators
	/// These arguments are hidden to ensure shape is correct
	/// since meta data can influence shape
	std::tuple<ARGS...> meta_;
};

}

#endif // LLO_NODE_HPP
