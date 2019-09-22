///
/// data.hpp
/// pbm
///
/// Purpose:
/// Define functions for marshaling equation graph
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
using TensT = std::vector<teq::TensptrT>;

/// String list type used for paths
using StringsT = std::list<fmts::string>;

struct iSaver
{
	virtual std::string save_leaf (teq::iLeaf* leaf) = 0;

	virtual std::vector<double> save_shaper (const teq::CoordptrT& mapper) = 0;

	virtual std::vector<double> save_coorder (const teq::CoordptrT& mapper) = 0;
};

struct iLoader
{
	virtual teq::TensptrT generate_leaf (const char* data, teq::Shape shape,
		std::string typelabel, std::string label, bool is_const) = 0;

	virtual teq::TensptrT generate_func (std::string opname, teq::ArgsT args) = 0;

	virtual teq::CoordptrT generate_shaper (std::vector<double> coord) = 0;

	virtual teq::CoordptrT generate_coorder (
		std::string opname, std::vector<double> coord) = 0;
};

}

#endif // PBM_COMMON_HPP
