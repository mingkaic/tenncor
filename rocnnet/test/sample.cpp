#include <iostream>
#include <fstream>

#include "dbg/ade.hpp"

#include "llo/shear.hpp"

#include "pbm/graph.hpp"

#include "rocnnet/eqns/activations.hpp"

#include "rocnnet/modl/gd_trainer.hpp"

int main (int argc, char** argv)
{
	GOOGLE_PROTOBUF_VERIFY_VERSION;
	std::string outdir = "/tmp";
	std::string serialname = "gd_test.pbx";
	std::string serialpath = outdir + "/" + serialname;

	uint8_t n_in = 10;
	uint8_t n_out = n_in / 2;
	std::vector<LayerInfo> hiddens = {
		// use same sigmoid in static memory once models copy is established
		LayerInfo{9, sigmoid},
		LayerInfo{n_out, sigmoid}
	};
	MLP brain(n_in, hiddens, "brain");

	uint8_t n_batch = 3;
	ApproxFuncT approx = [](llo::DataNode& root, std::vector<llo::DataNode> leaves)
	{
		return sgd(root, leaves, 0.9); // learning rate = 0.9
	};
	GDTrainer trainer(brain, approx, n_batch, "gdn");

	auto vars = brain.get_variables();
	// trainer.train_in_ = std::vector<double>{
	// 	1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
	// 	1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
	// 	1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	llo::DataNode root = trainer.error_;

	for (auto var : vars)
	{
		auto der = root.derive(var);
		llo::GenericData gdata = llo::zero_prune(der).data(llo::DOUBLE);
		double* gdptr = (double*) gdata.data_.get();
		std::cout << ade::to_string(gdptr, gdptr + gdata.shape_.n_elems()) << std::endl;
	}

	llo::GenericData data = root.data(llo::DOUBLE);
	double* dptr = (double*) data.data_.get();
	std::cout << ade::to_string(dptr, dptr + data.shape_.n_elems()) << std::endl;


	tenncor::Graph graph;
	std::vector<llo::DataNode> roots = {trainer.error_};
	save_graph(graph, roots);
	std::fstream outstr(serialpath, std::ios::out | std::ios::trunc | std::ios::binary);
	if (!graph.SerializeToOstream(&outstr))
	{
		ade::warn("failed to serialize initial trainer");
	}

	google::protobuf::ShutdownProtobufLibrary();

	return 0;
}
