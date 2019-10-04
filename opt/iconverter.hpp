///
/// iconverter.hpp
/// opt
///
/// Purpose:
/// Define converter interface for building TEQ graphs given candidate IR graph
///

extern "C" {
#include "opt/parse/def.h"
}

#include "opt/candidate.hpp"

#ifndef OPT_ICONVERTER_HPP
#define OPT_ICONVERTER_HPP

namespace opt
{

/// Converter interface for building TEQ graphs
struct iConverter
{
	virtual ~iConverter (void) = default;

	/// Return converted TEQ graph root given candidate context
	/// and expected output shape
	virtual teq::TensptrT build (
		const ContexT& ctx, teq::Shape outshape) const = 0;

	//// Return string representation of converter
	virtual std::string to_string (void) const = 0;
};

/// Smart pointer of converter
using ConvptrT = std::shared_ptr<iConverter>;

}

#endif // OPT_ICONVERTER_HPP
