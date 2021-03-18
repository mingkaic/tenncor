#include "dbg/profile/gexf.hpp"

#ifdef DBG_PROFILE_GEXF_HPP

namespace dbg
{

namespace profile
{

void gexf_write (const std::string& outfilename, const teq::TensT& roots)
{
	eigen::Device realdev(
		eigen::get_runtime(), std::numeric_limits<size_t>::max());
	ProfilerDevice device(realdev);

	teq::TensSetT rootset(roots.begin(), roots.end());
	teq::Evaluator().evaluate(device, rootset);

	GexfWriter writer(device);
	teq::multi_visit(writer, roots);

	writer.write(outfilename);
}

void gexf_write (std::ostream& os,
	const teq::TensT& roots, const std::string& outdir)
{
	eigen::Device realdev(
		eigen::get_runtime(), std::numeric_limits<size_t>::max());
	ProfilerDevice device(realdev);

	teq::TensSetT rootset(roots.begin(), roots.end());
	teq::Evaluator().evaluate(device, rootset);

	GexfWriter writer(device);
	teq::multi_visit(writer, roots);

	NodeDisplay display(writer.ids_, outdir);

	writer.write(os);
}

}

}

#endif // DBG_PROFILE_GRAPH_HPP
