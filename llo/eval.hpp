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
};

/// Interface for evaluating data of a type
struct iEvaluable
{
	virtual ~iEvaluable (void) = default;

	/// Return data evaluated from subgraph converted to input type
	virtual GenericData data (DTYPE dtype) const = 0;
};

/// Type used by context to associate ade::Tensors to Sources
using SourcePoolT = std::unordered_map<ade::Tensor*,std::shared_ptr<iSource>>;

/// Type used by context to associate ade::iFunctor to llo meta-data wrappers
using FuncPoolT = std::unordered_map<
	ade::iFunctor*,std::shared_ptr<iEvaluable>>;

/// Context used to associate ade nodes to llo nodes under a particular graph
struct EvalCtx final
{
	EvalCtx (void) = default;

	EvalCtx (ade::Tensor* srckey, std::shared_ptr<iSource>& srcval)
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
			funks_.insert(ctx->funks_.begin(), ctx->funks_.end());
		}
	}

	/// List all ade-source mapping
	SourcePoolT srcs_;

	/// List all ade-funcwrapper mapping
	FuncPoolT funks_;
};

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
	void visit (ade::Tensor* leaf) override;

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override;

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
	DataNode derive (ade::Tensorptr& wrt) const
	{
		ade::Tensorptr grad = tensor_->gradient(wrt);
		return DataNode(ctx_, grad);
	}

	/// Return DataNode of gradient tree derived with respect to wrt DataNode
	DataNode derive (DataNode& wrt) const
	{
		return derive(wrt.tensor_);
	}

	/// Accumulated context
	EvalCtx ctx_;

	/// Subgraph root
	ade::Tensorptr tensor_;
};

/// Evaluate the data of children for func according to inputs ctx and dtype
void get_func_children (std::vector<GenericData>& out,
	const EvalCtx& ctx, DTYPE dtype, ade::iFunctor* func);

}

#endif // LLO_EVAL_HPP
