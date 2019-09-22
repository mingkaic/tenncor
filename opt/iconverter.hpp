extern "C" {
#include "opt/parse/def.h"
}

#include "opt/candidate.hpp"

#ifndef OPT_ICONVERTER_HPP
#define OPT_ICONVERTER_HPP

namespace opt
{

struct iConverter
{
	virtual ~iConverter (void) = default;

	virtual teq::TensptrT build (
		const ContexT& ctx, teq::Shape outshape) const = 0;

	virtual std::string to_string (void) const = 0;
};

using ConvptrT = std::shared_ptr<iConverter>;

}

#endif // OPT_ICONVERTER_HPP
