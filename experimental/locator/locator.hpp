#include "teq/ifunctor.hpp"

#include "tag/tag.hpp"

#ifndef TAG_LOCATOR_HPP
#define TAG_LOCATOR_HPP

namespace tag
{

std::string display_location (teq::iTensor* tens,
	teq::TensT known_roots = {},
	TagRegistry& tagreg = tag::get_reg());

std::string display_location (teq::TensptrT tens,
	teq::TensptrsT known_roots = {},
	TagRegistry& tagreg = tag::get_reg());

}

#endif // TAG_LOCATOR_HPP
