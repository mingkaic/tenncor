#include "dbg/profile/graph.hpp"

#ifdef DBG_PROFILE_GRAPH_HPP

namespace dbg
{

namespace profile
{

void gexf_write (const std::string& outfilename, const teq::TensT& roots)
{
    eigen::Device realdev(eigen::get_runtime(), std::numeric_limits<size_t>::max());
    ProfilerDevice device(realdev);

    teq::TensSetT rootset(roots.begin(), roots.end());
    teq::Evaluator().evaluate(device, rootset);

    GexfWriter writer(device);
    teq::multi_visit(writer, roots);

    writer.write(outfilename);
}

}

}

#endif // DBG_PROFILE_GRAPH_HPP
