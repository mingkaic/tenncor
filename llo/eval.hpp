///
/// eval.hpp
/// llo
///
/// Purpose:
/// Define evaluation interface for calculating data
///

#include <unordered_map>

#include "ade/functor.hpp"

#include "llo/opmap.hpp"

#ifndef LLO_EVAL_HPP
#define LLO_EVAL_HPP

namespace llo
{

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

	/// Return internal tensor referencing this
	virtual const std::shared_ptr<ade::Tensor>& inner (void) const = 0;
};

/// Type used by context to associate ade::Tensors to Sources
using SourcePoolT = std::unordered_map<ade::iTensor*,std::shared_ptr<iSource>>;

/// Context used to associate ade nodes to llo nodes under a particular graph
struct EvalCtx final
{
	EvalCtx (void) = default;

	EvalCtx (ade::iTensor* srckey, std::shared_ptr<iSource>& srcval)
	{
		if (srcval != nullptr && srckey != nullptr)
		{
			srcs_[srckey] = srcval;
		}
	}

	EvalCtx (std::vector<const EvalCtx*> contexas)
	{
		for (const EvalCtx* ctx : contexas)
		{
			srcs_.insert(ctx->srcs_.begin(), ctx->srcs_.end());
		}
	}

	/// List all ade-source mapping
	SourcePoolT srcs_;
};

/// Evaluate the data of children for func according to inputs ctx and dtype
void calc_func_args (DataArgsT& out, const EvalCtx& ctx,
	DTYPE dtype, ade::iFunctor* func);

/// Visitor implementation to evaluate ade nodes according to ctx and dtype
/// Given a global context containing ade-llo association maps, get data from
/// llo::Sources when possible, otherwise treat native ade::Tensors as zeroes
/// Additionally, Evaluator attempts to get meta-data from llo::FuncWrapper
/// before checking native ade::Functor
struct Evaluator final : public ade::iTraveler
{
	Evaluator (const EvalCtx& ctx, DTYPE dtype) :
		ctx_(&ctx), dtype_(dtype) {}

	/// Implementation of iTraveler
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

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		ade::OPCODE opcode = func->get_code();
		out_ = GenericData(func->shape(), dtype_);

		DataArgsT argdata;
		calc_func_args(argdata, *ctx_, dtype_, func);
		op_exec(opcode, out_, argdata);
	}

	/// Output data evaluated upon visiting node
	GenericData out_;

private:
	/// Context used when evaluating node
	const EvalCtx* ctx_; // not owner

	/// Output type when evaluating data
	DTYPE dtype_;
};

/// API Node encapsulating ade::Tensorptr and context
/// Context maps tensors to data sources/meta-data evaluators in this subgraph
struct DataNode
{
	DataNode (EvalCtx ctx, ade::Tensorptr tensor) :
		ctx_(ctx), tensor_(tensor) {}

	virtual ~DataNode (void) = default;

	/// Return data according to accumulated context and output type
	GenericData data (DTYPE dtype)
	{
		Evaluator eval(ctx_, dtype);
		tensor_->accept(eval);
		return eval.out_;
	}

	/// Return DataNode of gradient tree derived with respect to wrt tensor
	DataNode derive (const ade::iTensor* wrt)
	{
		ade::Tensorptr grad = tensor_->gradient(wrt);
		return DataNode(ctx_, grad);
	}

	/// Return DataNode of gradient tree derived with respect to wrt DataNode
	DataNode derive (DataNode& wrt)
	{
		return derive(wrt.tensor_.get());
	}

	/// Return iSource mapped by tensor_ if found in ctx_, otherwise null
	std::shared_ptr<llo::iSource> source (void) const
	{
		return get_src(tensor_);
	}

	/// Return iSource mapped by target if found in ctx_, otherwise null
	std::shared_ptr<llo::iSource> get_src (ade::Tensorptr target) const
	{
		std::shared_ptr<llo::iSource> out = nullptr;
		auto it = ctx_.srcs_.find(target.get());
		if (ctx_.srcs_.end() != it)
		{
			out = it->second;
		}
		return out;
	}

	/// Accumulated context
	EvalCtx ctx_;

	/// Subgraph root
	ade::Tensorptr tensor_;
};

}

#endif // LLO_EVAL_HPP
