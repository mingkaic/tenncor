#include "opt/matcher.hpp"
#include "opt/iconverter.hpp"

#ifndef OPT_OPTIMIZE_HPP
#define OPT_OPTIMIZE_HPP

namespace opt
{

struct OptCtx
{
	VoterPool voters_;

	ConvptrT const_conv_;

	std::unordered_map<std::string,ConvptrT> converts_;
};

ade::TensT optimize (ade::TensT roots, const OptCtx& opts);

}

#endif // OPT_OPTIMIZE_HPP
