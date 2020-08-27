
#ifndef DISTRIB_PRINT_DATA_HPP
#define DISTRIB_PRINT_DATA_HPP

#include "dbg/print/print.hpp"

#include "tenncor/distrib/reference.hpp"

namespace distr
{

namespace print
{

struct AsciiRemote
{
	std::string refid_;

	std::string clusterid_;

	std::string prefix_;
};

using AsciiRemotesT = std::vector<AsciiRemote>;

/// Caching entry for ListAscii responses
/// Maintains remote information
struct AsciiTemplate
{
	AsciiTemplate (std::string format, std::vector<AsciiRemote> remotes) :
		remotes_(remotes)
	{
		format_ << format;
	}

	AsciiTemplate (teq::iTensor* tens, const PrintEqConfig& cfg);

	std::stringstream format_;

	AsciiRemotesT remotes_;
};

/// Cache for Ascii Templates per Root UUID
struct DistrPrintData
{
	// helper function to stitch together ascii template using values in cache
	void stitch_ascii (std::ostream& os, AsciiTemplate& ascii,
		const std::string& prefix, const std::string& first_line_prefix);

	types::StrUMapT<AsciiTemplate> remote_templates_;
};

}

}

#endif // DISTRIB_PRINT_DATA_HPP
