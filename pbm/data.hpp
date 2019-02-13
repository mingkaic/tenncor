///
/// data.hpp
/// pbm
///
/// Purpose:
/// Define functions for marshaling equation graph
///

#include <list>

#include "ade/ade.hpp"

#include "pbm/graph.pb.h"

#ifndef PBM_COMMON_HPP
#define PBM_COMMON_HPP

namespace pbm
{

/// Tensptr vector type
using TensT = std::vector<ade::TensptrT>;

/// String list type used for paths
using StringsT = std::list<std::string>;

struct iSaver
{
	virtual std::string save_leaf (bool& is_const, ade::iLeaf* leaf) = 0;

	virtual std::vector<double> save_shaper (const ade::CoordptrT& mapper) = 0;

	virtual std::vector<double> save_coorder (const ade::CoordptrT& mapper) = 0;
};

struct iLoader
{
	virtual ade::TensptrT generate_leaf (const char* data, ade::Shape shape,
		size_t typecode, std::string label, bool is_const) = 0;

	virtual ade::TensptrT generate_func (ade::Opcode opcode, ade::ArgsT args) = 0;

	virtual ade::CoordptrT generate_shaper (std::vector<double> coord) = 0;

	virtual ade::CoordptrT generate_coorder (
		ade::Opcode opcode, std::vector<double> coord) = 0;
};

}

#endif // PBM_COMMON_HPP
