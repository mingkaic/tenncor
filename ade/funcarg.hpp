///
/// funcarg.hpp
/// ade
///
/// Purpose:
/// Define functor argument wrapper to carryover shape and coordinate mappers
///

#include "ade/itensor.hpp"
#include "ade/coord.hpp"

#ifndef ADE_FUNCARG_HPP
#define ADE_FUNCARG_HPP

namespace ade
{

/// Coordinate mapper and tensor pair
struct FuncArg final
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

	/// Return shape of tensor filtered through coordinate mapper
	Shape shape (void) const
	{
		ade::Shape shape = tensor_->shape();
		CoordT out;
		CoordT in;
		std::copy(shape.begin(), shape.end(), in.begin());
		shaper_->forward(out.begin(), in.begin());
		std::vector<DimT> slist(rank_cap);
		std::transform(out.begin(), out.end(), slist.begin(),
			[](CDimT cd) -> DimT
			{
				if (cd < 0)
				{
					cd = -cd - 1;
				}
				return std::round(cd);
			});
		return Shape(slist);
	}

	/// Return tensor being mapped
	TensptrT get_tensor (void) const
	{
		return tensor_;
	}

	/// Return shaper coord map
	CoordptrT get_shaper (void) const
	{
		return shaper_;
	}

	/// Return map_io_ flag, True if coorder accepts input coord
	/// and generated output, False otherwise
	bool map_io (void) const
	{
		return map_io_;
	}

	/// Return coord map for coordinates
	CoordptrT get_coorder (void) const
	{
		return coorder_;
	}

private:
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

/// Vector representation of ade tensor pointers
using TensT = std::vector<TensptrT>;

/// Return FuncArg that identity maps input tensor
FuncArg identity_map (TensptrT tensor);

/// Return FuncArg that reduces input tensor at
/// rank then snip the dimension at rank
/// E.g.: tensor w/ shape [2, 3, 4], rank = 1 gets mapped to [2, 4]
FuncArg reduce_1d_map (TensptrT tensor, uint8_t rank);

/// Return FuncArg that reduces input tensor by
/// units in reduction vector after specified rank
/// E.g.: tensor w/ shape [2, 3, 4], rank = 0, red = [2, 3]
/// gets mapped to [1, 1, 4]
FuncArg reduce_map (TensptrT tensor,
	uint8_t rank, std::vector<DimT> red);

/// Return FuncArg that extends input tensor by
/// rank and extension vector
/// E.g.: tensor w/ shape [2, 1, 1], rank = 1, red = [3, 4]
/// gets mapped to [2, 3, 4]
FuncArg extend_map (TensptrT tensor,
	uint8_t rank, std::vector<DimT> ext);

/// Return FuncArg that permutes input tensor by order
/// E.g.: tensor w/ shape [2, 3, 4], order = [1, 2, 0]
/// gets mapped to [3, 4, 2]
FuncArg permute_map (TensptrT tensor, std::vector<uint8_t> order);

/// Return FuncArg that flips input tensor along dimension
FuncArg flip_map (TensptrT tensor, uint8_t dim);

/// Return ArgsT with each tensor in TensT attached to identity mapper
ArgsT to_args (TensT tens);

}

#endif // ADE_FUNCARG_HPP
