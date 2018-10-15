///
/// node.hpp
/// llo
///
/// Purpose:
/// Extend ade::iTensor with proxies that carry and evaluate tensor data
///

#include <unordered_map>

#include "ade/functor.hpp"

#include "llo/opmap.hpp"

#ifndef LLO_NODE_HPP
#define LLO_NODE_HPP

namespace llo
{

/// Interface for evaluating data of a type
struct iEvaluable
{
	virtual ~iEvaluable (void) = default;

	/// Return data evaluated from subgraph converted to input type
	virtual GenericData data (DTYPE dtype) const = 0;
};

/// Interface for leaves with tensor data
struct iSource
{
	virtual ~iSource (void) = default;

	/// Return data converted to input type
	virtual GenericData data (DTYPE dtype) const = 0;

	/// Return the type of data stored
	virtual DTYPE native_type (void) const = 0;

	/// Assign new data values
	virtual void reassign (const GenericRef& data) = 0;
};

using SourcePoolT = std::unordered_map<ade::Tensor*,std::shared_ptr<iSource>>;
using FuncPoolT = std::unordered_map<ade::iFunctor*,std::shared_ptr<iEvaluable>>;

struct EvalCtx final
{
	EvalCtx (ade::Tensor* srckey, std::shared_ptr<iSource>& srcval)
	{
		srcs_[srckey] = srcval;
	}

	EvalCtx (std::vector<const EvalCtx*> contexas)
	{
		for (const EvalCtx* ctx : contexas)
		{
			srcs_.insert(ctx->srcs_.begin(), ctx->srcs_.end());
			funks_.insert(ctx->funks_.begin(), ctx->funks_.end());
		}
	}

	SourcePoolT srcs_;
	FuncPoolT funks_;
};

void fill_one (char* cptr, size_t n, DTYPE dtype);

void get_func_children (std::vector<GenericData>& out,
	const EvalCtx& ctx, DTYPE dtype, ade::iFunctor* func);

struct Evaluator final : public ade::Traveler
{
	Evaluator (const EvalCtx& ctx, DTYPE dtype) :
		ctx_(&ctx), dtype_(dtype) {}

	void visit (ade::Tensor* leaf) override
	{
		if (leaf == ade::Tensor::SYMBOLIC_ONE.get())
		{
			out_ = GenericData(ade::Shape(), dtype_);
			fill_one(out_.data_.get(), 1, dtype_);
			return;
		}
		auto srcpair = ctx_->srcs_.find(leaf);
		if (ctx_->srcs_.end() != srcpair)
		{
			out_ = srcpair->second->data(dtype_);
		}
		else
		{
			if (leaf != ade::Tensor::SYMBOLIC_ZERO.get())
			{
				ade::warnf("evaluating an ade::Tensor %s without associated "
					"Source according to input context... treating data as 0",
					leaf->to_string().c_str()); // todo: describe ctx for comprehensive report
			}
			out_ = GenericData(ade::Shape(), dtype_);
			std::memset(out_.data_.get(), 0, type_size(dtype_));
		}
	}

	void visit (ade::iFunctor* func) override
	{
		auto funkpair = ctx_->funks_.find(func);
		if (ctx_->funks_.end() != funkpair)
		{
			out_ = funkpair->second->data(dtype_);
			return;
		}
		// else visit pure ade::iFunctor
		ade::OPCODE opcode = func->get_code();

		out_ = GenericData(func->shape(), dtype_);

		std::vector<GenericData> argdata;
		get_func_children(argdata, *ctx_, dtype_, func);
		// todo: think of better way to deal with these metadata
		// (perhaps pass around as metadata interface of template implementation)
		switch (opcode)
		{
			case ade::MATMUL:
			{
				if (auto mf = static_cast<ade::Functor<
					ade::MATMUL,uint8_t,uint8_t>*>(func))
				{
					op_exec(opcode, out_, argdata,
						std::get<0>(mf->meta()), std::get<1>(mf->meta()));
				}
			}
			break;
			case ade::PERMUTE:
			{
				auto pf = static_cast<ade::Functor<
					ade::PERMUTE,std::vector<uint8_t>>*>(func);
				op_exec(opcode, out_, argdata, std::get<0>(pf->meta()));
			}
			break;
			case ade::EXTEND:
			{
				auto ef = static_cast<ade::Functor<
					ade::EXTEND,std::vector<ade::DimT>>*>(func);
				op_exec(opcode, out_, argdata, std::get<0>(ef->meta()));
			}
			break;
			case ade::RESHAPE:
			{
				auto rf = static_cast<ade::Functor<
					ade::RESHAPE,std::vector<ade::DimT>>*>(func);
				op_exec(opcode, out_, argdata, std::get<0>(rf->meta()));
			}
			break;
			default:
				op_exec(opcode, out_, argdata);
		}
	}

	GenericData out_;

private:
	const EvalCtx* ctx_; // not owner

	DTYPE dtype_;
};

/// API Node encapsulating ade::Tensorptr and context
/// Context maps tensors to data sources/meta-data evaluators in this subgraph
struct DataNode
{
	DataNode (EvalCtx ctx, ade::Tensorptr tensor) :
		ctx_(ctx), tensor_(tensor) {}

	virtual ~DataNode (void) = default;

	GenericData data (DTYPE dtype)
	{
		Evaluator eval(ctx_, dtype);
		tensor_->accept(eval);
		return eval.out_;
	}

	DataNode derive (ade::Tensorptr& wrt)
	{
		ade::Tensorptr grad = tensor_->gradient(wrt);
		return DataNode(ctx_, grad);
	}

	DataNode derive (DataNode& wrt)
	{
		return derive(wrt.tensor_);
	}

	EvalCtx ctx_;

	ade::Tensorptr tensor_;
};

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

private:
	Source (std::shared_ptr<ade::Tensor>& tensor, std::vector<T>& data) :
		tensor_(tensor), data_(data) {}

	/// Tensor node of subgraph
	std::shared_ptr<ade::Tensor> tensor_;

	/// Tensor data
	std::vector<T> data_;
};

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
