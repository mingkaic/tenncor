#include <fstream>
#include <sstream>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"

#include "rocnnet/eqns/activations.hpp"

#include "rocnnet/modl/fcon.hpp"
#include "rocnnet/modl/mlp.hpp"
// #include "rocnnet/modl/rbm.hpp"
// #include "rocnnet/modl/dbn.hpp"
// #include "rocnnet/modl/conv.hpp"

#include "rocnnet/trainer/mlp_trainer.hpp"
#include "rocnnet/trainer/dqn_trainer.hpp"
// #include "rocnnet/trainer/rbm_trainer.hpp"
// #include "rocnnet/trainer/dbn_trainer.hpp"

namespace py = pybind11;

namespace pyrocnnet
{

modl::LayerInfo layerinfo_init (modl::HiddenFunc hidden, size_t n_out)
{
	return modl::LayerInfo{n_out, hidden};
}

DQNInfo dqninfo_init (size_t train_interval = 5,
	double rand_action_prob = 0.05,
	double discount_rate = 0.95,
	double target_update_rate = 0.01,
	double exploration_period = 1000,
	size_t store_interval = 5,
	uint8_t mini_batch_size = 32,
	size_t max_exp = 30000)
{
	return DQNInfo{
		train_interval,
		rand_action_prob,
		discount_rate,
		target_update_rate,
		exploration_period,
		store_interval,
		mini_batch_size,
		max_exp
	};
}

modl::FConptrT fcon_init (std::vector<uint8_t> n_inputs, size_t n_output,
	std::string label)
{
	return std::make_shared<modl::FCon>(n_inputs, n_output, label);
}

modl::MLPptrT mlp_init (size_t n_input, std::vector<modl::LayerInfo> layers,
	std::string label)
{
	return std::make_shared<modl::MLP>(n_input, layers, label);
}

// modl::RBMptrT rbm_init (size_t n_input, size_t n_hidden, std::string label)
// {
// 	return std::make_shared<modl::RBM>(n_input, n_hidden, label);
// }

// modl::DBNptrT dbn_init (size_t n_input, std::vector<size_t> n_hiddens,
// 	std::string label)
// {
// 	return std::make_shared<modl::DBN>(n_input,
// 		std::vector<uint8_t>(n_hiddens.begin(), n_hiddens.end()), label);
// }

eqns::ApproxFuncT get_sgd (double learning_rate)
{
	return [=](ead::NodeptrT<double>& root, eqns::VariablesT leaves)
	{
		return eqns::sgd(root, leaves, learning_rate);
	};
}

eqns::ApproxFuncT get_rms_momentum (double learning_rate,
	double discount_factor, double epsilon)
{
	return [=](ead::NodeptrT<double>& root, eqns::VariablesT leaves)
	{
		return eqns::rms_momentum(root, leaves, learning_rate,
			discount_factor, epsilon);
	};
}

}

PYBIND11_MODULE(rocnnet, m)
{
	m.doc() = "rocnnet api";

	// common parent
	py::class_<modl::iMarshaler,modl::MarsptrT> marshaler(m, "Marshaler");

	// models
	py::class_<modl::FCon,modl::iMarshaler,modl::FConptrT> fcon(m, "FCon");
	py::class_<modl::MLP,modl::iMarshaler,modl::MLPptrT> mlp(m, "MLP");
	// py::class_<modl::RBM,modl::iMarshaler,modl::RBMptrT> rbm(m, "RBM");
	// py::class_<modl::DBN,modl::iMarshaler,modl::DBNptrT> dbn(m, "DBN");

	// support classes
	py::class_<eqns::VarAssign> assigns(m, "VarAssign");

	py::class_<modl::LayerInfo> layerinfo(m, "LayerInfo");
	py::class_<DQNInfo> dqninfo(m, "DQNInfo");
	py::class_<MLPTrainer> mlptrainer(m, "MLPTrainer");
	py::class_<DQNTrainer> dqntrainer(m, "DQNTrainer");
	// py::class_<RBMTrainer> rbmtrainer(m, "RBMTrainer");

	// marshaler
	marshaler
		.def("serialize_to_file", [](py::object self, ead::NodeptrT<double> source,
			std::string filename)
		{
			std::fstream output(filename,
				std::ios::out | std::ios::trunc | std::ios::binary);
			if (false == modl::save(output, source->get_tensor(),
				self.cast<modl::iMarshaler*>()))
			{
				logs::errorf("cannot save to file %s", filename.c_str());
			}
		}, "load a version of this instance from a data")
		.def("serialize_to_string", [](py::object self, ead::NodeptrT<double> source)
		{
			std::stringstream savestr;
			modl::save(savestr, source->get_tensor(),
				self.cast<modl::iMarshaler*>());
			return savestr.str();
		}, "load a version of this instance from a data")
		.def("parse_from_string", [](py::object self, std::string data)
		{
			modl::MarsptrT out(self.cast<modl::iMarshaler*>()->clone());
			std::stringstream loadstr;
			loadstr << data;
			modl::load(loadstr, out.get());
			return out;
		}, "load a version of this instance from a data");

	// fcon
	m.def("get_fcon", &pyrocnnet::fcon_init);
	fcon
		.def("copy", [](py::object self)
		{
			return std::make_shared<modl::FCon>(*self.cast<modl::FCon*>());
		}, "deep copy this instance")
		.def("forward", [](py::object self, ead::NodesT<double> inputs)
		{
			return (*self.cast<modl::FCon*>())(inputs);
		}, "forward input tensor and returned connected output");

	// mlp
	m.def("get_layer", &pyrocnnet::layerinfo_init);
	m.def("get_mlp", &pyrocnnet::mlp_init);
	mlp
		.def("copy", [](py::object self)
		{
			return std::make_shared<modl::MLP>(*self.cast<modl::MLP*>());
		}, "deep copy this instance")
		.def("forward", [](py::object self, ead::NodeptrT<double> input)
		{
			return (*self.cast<modl::MLP*>())(input);
		}, "forward input tensor and returned connected output");

	// // rbm
	// m.def("get_rbm", &pyrocnnet::rbm_init);
	// rbm
	// 	.def("copy", [](py::object self)
	// 	{
	// 		return std::make_shared<modl::RBM>(*self.cast<modl::RBM*>());
	// 	}, "deep copy this instance")
	// 	.def("forward", [](py::object self, ead::NodeptrT<double> input)
	// 	{
	// 		return self.cast<modl::RBM*>()->prop_up(input);
	// 	}, "forward input tensor and returned connected output")
	// 	.def("backward", [](py::object self, ead::NodeptrT<double> hidden)
	// 	{
	// 		return self.cast<modl::RBM*>()->prop_down(hidden);
	// 	}, "backward hidden tensor and returned connected output")
	// 	.def("reconstruct_visible", [](py::object self, ead::NodeptrT<double> input)
	// 	{
	// 		return self.cast<modl::RBM*>()->reconstruct_visible(input);
	// 	}, "reconstruct input")
	// 	.def("reconstruct_hidden", [](py::object self, ead::NodeptrT<double> hidden)
	// 	{
	// 		return self.cast<modl::RBM*>()->reconstruct_hidden(hidden);
	// 	}, "reconstruct output");

	// // dbn
	// m.def("get_dbn", &pyrocnnet::dbn_init);
	// dbn
	// 	.def("copy", [](py::object self)
	// 	{
	// 		return std::make_shared<modl::DBN>(*self.cast<modl::DBN*>());
	// 	}, "deep copy this instance")
	// 	.def("forward", [](py::object self, ead::NodeptrT<double> input)
	// 	{
	// 		return (*self.cast<modl::DBN*>())(input);
	// 	}, "forward input tensor and returned connected output");


	// mlptrainer
	mlptrainer
		.def(py::init<modl::MLPptrT,ead::Session<double>&,eqns::ApproxFuncT,uint8_t>())
		.def("train", &MLPTrainer::train, "train internal variables")
		.def("train_in", [](py::object self)
		{
			return self.cast<MLPTrainer*>()->train_in_;
		}, "get train_in variable")
		.def("expected_out", [](py::object self)
		{
			return self.cast<MLPTrainer*>()->expected_out_;
		}, "get expected_out variable")
		.def("train_out", [](py::object self)
		{
			return self.cast<MLPTrainer*>()->train_out_;
		}, "get training node")
		.def("error", [](py::object self)
		{
			return self.cast<MLPTrainer*>()->error_;
		}, "get error node")
		.def("brain", [](py::object self)
		{
			return self.cast<MLPTrainer*>()->brain_;
		}, "get mlp");

	// dqntrainer
	m.def("get_dqninfo", &pyrocnnet::dqninfo_init,
		py::arg("train_interval") = 5,
		py::arg("rand_action_prob") = 0.05,
		py::arg("discount_rate") = 0.95,
		py::arg("target_update_rate") = 0.01,
		py::arg("exploration_period") = 1000,
		py::arg("store_interval") = 5,
		py::arg("mini_batch_size") = 32,
		py::arg("max_exp") = 30000);
	dqntrainer
		.def(py::init<modl::MLPptrT,ead::Session<double>&,eqns::ApproxFuncT,DQNInfo>())
		.def("action", &DQNTrainer::action, "get next action")
		.def("store", &DQNTrainer::store, "save observation, action, and reward")
		.def("train", &DQNTrainer::train, "train qnets")
		.def("error", &DQNTrainer::get_error, "get prediction error")
		.def("ntrained", &DQNTrainer::get_numtrained, "get number of iterations trained")
		.def("train_out", [](py::object self)
		{
			return self.cast<DQNTrainer*>()->train_out_;
		}, "get training node");

	// // rbmtrainer
	// rbmtrainer
	// 	.def(py::init<modl::RBMptrT,llo::VarptrT<double>,uint8_t,double,size_t>(),
	// 		py::arg("brain"), py::arg("persistent"), py::arg("batch_size"),
	// 		py::arg("learning_rate") = 1e-3, py::arg("n_cont_div") = 1)
	// 	.def("train", &RBMTrainer::train, "train internal variables")
	// 	.def("train_in", [](py::object self)
	// 	{
	// 		return self.cast<RBMTrainer*>()->train_in_;
	// 	}, "get train_in variable")
	// 	.def("cost", [](py::object self)
	// 	{
	// 		return self.cast<RBMTrainer*>()->cost_;
	// 	}, "get cost node")
	// 	.def("monitoring_cost", [](py::object self)
	// 	{
	// 		return self.cast<RBMTrainer*>()->monitoring_cost_;
	// 	}, "get monitoring cost node")
	// 	.def("brain", [](py::object self)
	// 	{
	// 		return self.cast<RBMTrainer*>()->brain_;
	// 	}, "get rbm");


	// inlines
	m.def("identity", [](ead::NodeptrT<double> x) { return x; });
	m.def("sigmoid", &eqns::sigmoid);
	m.def("slow_sigmoid", &eqns::slow_sigmoid);
	m.def("tanh", &eqns::tanh);
	m.def("softmax", &eqns::softmax);

	m.def("get_sgd", &pyrocnnet::get_sgd,
		py::arg("learning_rate") = 0.5);
	m.def("get_rms_momentum", &pyrocnnet::get_rms_momentum,
		py::arg("learning_rate") = 0.5,
		py::arg("discount_factor") = 0.99,
		py::arg("epsilon") = std::numeric_limits<double>::epsilon());
};
