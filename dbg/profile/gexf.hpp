#include "dbg/profile/device.hpp"
#include "dbg/profile/streamwriter.hpp"
#include "dbg/profile/nodedisplay.hpp"

#ifndef DBG_PROFILE_GEXF_HPP
#define DBG_PROFILE_GEXF_HPP

namespace dbg
{

namespace profile
{

static const std::string stat_attr = "runtime_ns";

static const std::string usage_attr = "node_usage";

static const std::string shape_attr = "node_shape";

static const std::string density_attr = "density";

struct GexfWriter final : public teq::iOnceTraveler
{
	GexfWriter (ProfilerDevice& dev) : dev_(&dev),
		graph_(&gexf_->getDirectedGraph()),
		data_(&gexf_->getData())
	{
		data_->addNodeAttributeColumn("0", stat_attr, "DOUBLE");
		data_->setNodeAttributeDefault("0", "0.0");
		data_->addNodeAttributeColumn("1", usage_attr, "STRING");
		data_->setNodeAttributeDefault("1", "ANY");
		data_->addNodeAttributeColumn("2", shape_attr, "STRING");
		data_->setNodeAttributeDefault("2", "[]");
		data_->addNodeAttributeColumn("3", "density", "DOUBLE");
		data_->setNodeAttributeDefault("3", "1");
		data_->addEdgeAttributeColumn("4", "position", "INT");
		data_->setEdgeAttributeDefault("4", "0");
	}

	void write (std::ostream& os) const
	{
		libgexf::StreamWriter writer;
		writer.init(gexf_.get());
		writer.write(os);
	}

	void write (const std::string& filename) const
	{
		libgexf::FileWriter writer;
		writer.init(filename, gexf_.get());
		writer.write();
	}

	teq::TensMapT<std::string> ids_;

private:
	/// Implementation of iOnceTraveler
	void visit_leaf (teq::iLeaf& leaf) override
	{
		auto uuid = gen_->get_str();
		ids_.emplace(&leaf, uuid);
		graph_->addNode(uuid);
		data_->setNodeLabel(uuid, leaf.to_string());
		data_->setNodeValue(uuid, "1", teq::get_usage_name(leaf.get_usage()));
		data_->setNodeValue(uuid, "2", leaf.shape().to_string());
		data_->setNodeValue(uuid, "3",
			fmts::to_string(eigen::calc_density(leaf)));
	}

	/// Implementation of iOnceTraveler
	void visit_func (teq::iFunctor& func) override
	{
		auto uuid = gen_->get_str();
		ids_.emplace(&func, uuid);
		graph_->addNode(uuid);
		auto children = func.get_args();
		teq::multi_visit(*this, children);
		for (size_t i = 0, n = children.size(); i < n; ++i)
		{
			auto child = children[i];
			auto edge_id = gen_->get_str();
			auto child_id = estd::must_getf(ids_, child.get(),
				"cannot find id for %s", child->to_string().c_str());
			graph_->addEdge(edge_id, child_id, uuid);
			data_->setEdgeValue(edge_id, "4", fmts::to_string(i));
		}
		data_->setNodeLabel(uuid, func.to_string());
		auto stat = estd::must_getf(dev_->stats_, &func,
			"cannot find stats for %s", func.to_string().c_str());
		data_->setNodeValue(uuid, "0", fmts::to_string(stat));
		if (auto paint = dynamic_cast<marsh::String*>(func.get_attr("paint")))
		{
			data_->setNodeValue(uuid, "1", paint->to_string());
		}
		data_->setNodeValue(uuid, "2", func.shape().to_string());
		data_->setNodeValue(uuid, "3",
			fmts::to_string(eigen::calc_density(func)));
	}

	std::unique_ptr<libgexf::GEXF> gexf_ = std::make_unique<libgexf::GEXF>();

	ProfilerDevice* dev_;

	libgexf::DirectedGraph* graph_;

	libgexf::Data* data_;

	global::GenPtrT gen_ = global::get_generator();
};

#undef _TYPE_DISPLAY

void gexf_write (const std::string& outfilename, const teq::TensT& roots);

void gexf_write (std::ostream& outfilename, const teq::TensT& roots,
	const std::string& outdir = "/tmp");

}

}

#endif // DBG_PROFILE_GEXF_HPP
