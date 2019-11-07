///
/// funcarg.hpp
/// teq
///
/// Purpose:
/// Define functor argument wrapper to carryover shape and coordinate mappers
///

#include "teq/itensor.hpp"
#include "teq/coord.hpp"

#ifndef TEQ_FUNCARG_HPP
#define TEQ_FUNCARG_HPP

namespace teq
{

struct iFuncArg
{
	virtual ~iFuncArg (void) = default;

	/// Return this instance copied
	iFuncArg* clone (void) const
	{
		return clone_impl();
	}

	/// Return shape of tensor filtered through shaper
	virtual Shape shape (void) const = 0;

	/// Return shape of tensor argument
	virtual Shape argshape (void) const = 0;

	/// Return tensor being mapped
	virtual TensptrT get_tensor (void) const = 0;

	/// Return shape transformer from argument shape to desired function shape
	virtual CoordptrT get_shaper (void) const = 0;

	// todo: add getter for coordinate information

protected:
	virtual iFuncArg* clone_impl (void) const = 0;
};

using CstArgsT = std::vector<std::reference_wrapper<const iFuncArg>>;

Shape apply_shaper (const CoordptrT& shaper, Shape inshape);

/// Coordinate mapper and tensor pair
struct FuncArg final : public iFuncArg
{
	/// Construct FuncArg auto deducing coorder_ and map_io_ flag
	FuncArg (TensptrT tensor, CoordptrT shaper) :
		tensor_(tensor), shaper_(shaper)
	{
		if (tensor_ == nullptr)
		{
			logs::fatal("cannot map a null tensor");
		}
		map_io_ = tensor_->shape().n_elems() > shape().n_elems();
		if (shaper == identity || map_io_)
		{
			coorder_ = shaper;
		}
		else
		{
			coorder_ = CoordptrT(shaper->reverse());
		}
	}

	/// Construct FuncArg with specific coorder_ and map_io_ flag
	FuncArg (TensptrT tensor, CoordptrT shaper,
		bool map_io, CoordptrT coorder) :
		tensor_(tensor), shaper_(shaper),
		map_io_(map_io), coorder_(coorder)
	{
		if (tensor_ == nullptr)
		{
			logs::fatal("cannot map a null tensor");
		}
	}

	/// Implementation of iFuncArg
	Shape shape (void) const override
	{
		return apply_shaper(shaper_, tensor_->shape());
	}

	/// Implementation of iFuncArg
	Shape argshape (void) const override
	{
		return tensor_->shape();
	}

	/// Return tensor being mapped
	TensptrT get_tensor (void) const override
	{
		return tensor_;
	}

	/// Return shaper coord map
	CoordptrT get_shaper (void) const override
	{
		return shaper_;
	}

	/// Return coord map for coordinates
	CoordptrT get_coorder (void) const
	{
		return coorder_;
	}

	/// Return map_io_ flag, True if coorder accepts input coord
	/// and generated output, False otherwise
	bool map_io (void) const
	{
		return map_io_;
	}

private:
	iFuncArg* clone_impl (void) const override
	{
		return new FuncArg(tensor_, shaper_, map_io_, coorder_);
	}

	/// Tensor reference
	TensptrT tensor_;

	/// Shape mapper
	CoordptrT shaper_;

	/// True if map input coordinate to output, False otherwise
	/// (if n_elems of inputshape > n_elems of outputshape)
	bool map_io_;

	/// Coordinate mapper
	CoordptrT coorder_;
};

/// Type of functor arguments
using ArgsT = std::vector<FuncArg>;

/// Return FuncArg that identity maps input tensor
FuncArg identity_map (TensptrT tensor);

/// Return FuncArg that reduces input tensor at
/// rank then snip the dimension at rank
/// E.g.: tensor w/ shape [2, 3, 4], rank = 1 gets mapped to [2, 4]
FuncArg reduce_1d_map (TensptrT tensor, RankT rank);

/// Return FuncArg that reduces input tensor by
/// units in reduction vector after specified rank
/// E.g.: tensor w/ shape [2, 3, 4], rank = 0, red = [2, 3]
/// gets mapped to [1, 1, 4]
FuncArg reduce_map (TensptrT tensor,
	RankT rank, std::vector<DimT> red);

/// Return FuncArg that extends input tensor by
/// rank and extension vector
/// E.g.: tensor w/ shape [2, 1, 1], rank = 1, red = [3, 4]
/// gets mapped to [2, 3, 4]
FuncArg extend_map (TensptrT tensor,
	RankT rank, std::vector<DimT> ext);

/// Return FuncArg that permutes input tensor by order
/// E.g.: tensor w/ shape [2, 3, 4], order = [1, 2, 0]
/// gets mapped to [3, 4, 2]
FuncArg permute_map (TensptrT tensor, std::vector<RankT> order);

/// Return FuncArg that flips input tensor along dimension
FuncArg flip_map (TensptrT tensor, RankT dim);

/// Return ArgsT with each tensor in TensptrsT attached to identity mapper
ArgsT to_args (TensptrsT tens);

}

#endif // TEQ_FUNCARG_HPP
