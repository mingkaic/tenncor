#include "tag/tag.hpp"

#ifndef TAG_GROUP_HPP
#define TAG_GROUP_HPP

namespace tag
{

void recursive_group_tag (ade::TensrefT tens, std::string group,
	std::unordered_set<ade::iTensor*> stops);

}

#endif // TAG_GROUP_HPP
