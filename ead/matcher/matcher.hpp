#include "ead/matcher/gopt.hpp"
#include "ead/matcher/parse.hpp"

#ifndef OPT_MATCHER_HPP
#define OPT_MATCHER_HPP

namespace opt
{

GraphOpt config_opt (void)
{
	std::ifstream cfg("ead/cfg/optimizations");
	if (false == cfg.is_open())
	{
		logs::warn(
			"default configuration is not found, returning empty optimizer");
	}
	TransformsT transforms = parse_stream(cfg);
	return GraphOpt(transforms);
}

}

#endif // OPT_MATCHER_HPP
