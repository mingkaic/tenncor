
#include "dbg/print/printsvc/data.hpp"

#ifdef DISTRIB_PRINT_DATA_HPP

namespace distr
{

namespace print
{

static const std::string ascii_formatkey = "{xj]yq<";

AsciiTemplate::AsciiTemplate (teq::iTensor* tens, const PrintEqConfig& cfg)
{
	PrettyTree<teq::iTensor*> renderer(
		[](teq::iTensor*& root, size_t depth) -> teq::TensT
		{
			if (auto f = dynamic_cast<teq::iFunctor*>(root))
			{
				auto args = f->get_args();
				teq::TensT tens;
				tens.reserve(args.size());
				std::transform(args.begin(), args.end(),
					std::back_inserter(tens),
					[](teq::TensptrT child)
					{
						return child.get();
					});
				return tens;
			}
			return {};
		},
		[&](std::ostream& out, teq::iTensor*& root, const std::string& prefix)
		{
			if (nullptr == root)
			{
				return;
			}
			if (auto ref = dynamic_cast<iDistrRef*>(root))
			{
				out << ascii_formatkey;
				remotes_.push_back(AsciiRemote{ref->node_id(), ref->cluster_id(), prefix});
				return;
			}
			out << "(";
			ten_stream(out, root, prefix, cfg);
			out << ")";
		});
	renderer.node_wrap = {"", ""};
	renderer.print(format_, tens);
}

void DistrPrintData::stitch_ascii (std::ostream& os, AsciiTemplate& ascii,
	const std::string& prefix, const std::string& first_line_prefix)
{
	size_t i = 0, n = ascii.remotes_.size();
	auto format = ascii.format_.str();
	fmts::trim(format);
	auto lines = fmts::split(format, "\n");
	for (size_t lno = 0, nl = lines.size(); lno < nl; ++lno)
	{
		auto line = lines[lno];
		if (i < n)
		{
			auto formbegin = line.find(ascii_formatkey);
			if (formbegin != std::string::npos)
			{
				auto it = line.begin();
				std::string init(it, it + formbegin);
				auto& remote = ascii.remotes_[i];
				++i;
				if (estd::has(remote_templates_, remote.refid_))
				{
					stitch_ascii(os,
						remote_templates_.at(remote.refid_),
						prefix + remote.prefix_,
						prefix + init + "[" + remote.clusterid_ + "]:");
					continue;
				}
				else
				{
					line = "(?)";
				}
			}
		}
		// add special prefix for first line only
		if (lno == 0)
		{
			os << first_line_prefix;
		}
		else
		{
			os << prefix;
		}
		os << line << "\n";
	}
}

}

}

#endif
