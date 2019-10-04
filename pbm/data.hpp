///
/// data.hpp
/// pbm
///
/// Purpose:
/// Define interfaces for marshaling concrete TEQ equation graph
///

#include <list>

#include "teq/teq.hpp"

#include "tag/tag.hpp"

#include "pbm/graph.pb.h"

#ifndef PBM_COMMON_HPP
#define PBM_COMMON_HPP

namespace pbm
{

/// Tensptr vector type
using TensptrsT = std::vector<teq::TensptrT>;

/// String list type used for paths
using StringsT = std::list<fmts::string>;

/// Interface for saving implementations of leaves, shaper, and coorder
struct iSaver
{
	/// Return string serialization of leaf data
	virtual std::string save_leaf (teq::iLeaf* leaf) = 0;

	/// Return vector serialization of shape coordinate map
	virtual std::vector<double> save_shaper (const teq::CoordptrT& mapper) = 0;

	/// Return vector serialization of coordinate map
	virtual std::vector<double> save_coorder (const teq::CoordptrT& mapper) = 0;
};

/// Interface for building implementations of leaves, functors, shaper, and coorder
struct iLoader
{
	/// Return leaf given raw data, shape, 
	/// data type encoding, and other meta data
	virtual teq::TensptrT generate_leaf (const char* data, teq::Shape shape,
		std::string typelabel, std::string label, bool is_const) = 0;

	/// Return functor given operator name and TEQ arguments
	virtual teq::TensptrT generate_func (std::string opname, teq::ArgsT args) = 0;

	/// Return shape coordinate map given vector serialization output 
	/// by corresponding iSaver
	virtual teq::CoordptrT generate_shaper (std::vector<double> coord) = 0;

	/// Return coordinate map given vector serialization output 
	/// by corresponding iSaver
	virtual teq::CoordptrT generate_coorder (
		std::string opname, std::vector<double> coord) = 0;
};

}

#endif // PBM_COMMON_HPP
