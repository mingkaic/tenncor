#include "libgexf/libgexf.h"

#include "dbg/profile/device.hpp"

#ifndef DBG_PROFILE_GRAPH_HPP
#define DBG_PROFILE_GRAPH_HPP

namespace dbg
{

namespace profile
{

static const std::string stat_attr = "runtime_ns";

static const std::string usage_attr = "node_usage";

static const std::string shape_attr = "node_shape";

struct GexfWriter final : public teq::iTraveler
{
    GexfWriter (ProfilerDevice& dev) : dev_(&dev),
        graph_(&gexf_->getDirectedGraph()),
        data_(&gexf_->getData()),
        gen_(global::get_generator())
	{
		data_->addNodeAttributeColumn("0", stat_attr, "DOUBLE");
		data_->setNodeAttributeDefault("0", "0.0");
		data_->addNodeAttributeColumn("1", usage_attr, "STRING");
		data_->setNodeAttributeDefault("1", "ANY");
		data_->addNodeAttributeColumn("2", shape_attr, "STRING");
		data_->setNodeAttributeDefault("2", "[]");
	}

	/// Implementation of iTraveler
	void visit (teq::iLeaf& leaf) override
    {
        if (estd::has(ids_, &leaf))
        {
            return;
        }
        auto uuid = gen_->get_str();
        graph_->addNode(uuid);
        ids_.emplace(&leaf, uuid);
        data_->setNodeLabel(uuid, leaf.to_string());
        data_->setNodeValue(uuid, "1", teq::get_usage_name(leaf.get_usage()));
        data_->setNodeValue(uuid, "2", leaf.shape().to_string());
    }

	/// Implementation of iTraveler
	void visit (teq::iFunctor& func) override
    {
        if (estd::has(ids_, &func))
        {
            return;
        }
        auto uuid = gen_->get_str();
        graph_->addNode(uuid);
        ids_.emplace(&func, uuid);
        auto children = func.get_args();
        teq::multi_visit(*this, children);
        for (auto child : children)
        {
            auto edge_id = gen_->get_str();
            auto child_id = estd::must_getf(ids_, child.get(),
                "cannot find id for %s", child->to_string().c_str());
            graph_->addEdge(edge_id, child_id, uuid);
        }
        data_->setNodeLabel(uuid, func.to_string());
        auto stat = estd::must_getf(dev_->stats_, &func,
            "cannot find stats for %s", func.to_string().c_str());
        data_->setNodeValue(uuid, "0", fmts::to_string(stat));
        data_->setNodeValue(uuid, "2", func.shape().to_string());
    }

    void write (const std::string& filename) const
    {
        libgexf::FileWriter writer;
        writer.init(filename, gexf_.get());
        writer.write();
    }

private:
    std::unique_ptr<libgexf::GEXF> gexf_ = std::make_unique<libgexf::GEXF>();

    ProfilerDevice* dev_;

    libgexf::DirectedGraph* graph_;

    libgexf::Data* data_;

    global::GenPtrT gen_;

    teq::TensMapT<std::string> ids_;
};

void gexf_write (const std::string& outfilename, const teq::TensT& roots);

}

}

#endif // DBG_PROFILE_GRAPH_HPP
