#include <fstream>
#include <sstream>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"

#include "ead/generated/pyapi.hpp"

#include "rocnnet/eqns/init.hpp"

#include "rocnnet/modl/activations.hpp"
#include "rocnnet/modl/dense.hpp"
#include "rocnnet/modl/rbm.hpp"
#include "rocnnet/modl/model.hpp"
// #include "rocnnet/modl/dbn.hpp"
// #include "rocnnet/modl/conv.hpp"

#include "rocnnet/trainer/mlp_trainer.hpp"
#include "rocnnet/trainer/dqn_trainer.hpp"
// #include "rocnnet/trainer/rbm_trainer.hpp"
// #include "rocnnet/trainer/dbn_trainer.hpp"

namespace py = pybind11;

namespace pyrocnnet
{

ade::Shape p2cshape (std::vector<py::ssize_t>& pyshape)
{
	return ade::Shape(std::vector<ade::DimT>(
		pyshape.rbegin(), pyshape.rend()));
}

// modl::DBNptrT dbn_init (size_t n_input, std::vector<size_t> n_hiddens,
// 	std::string label)
// {
// 	return std::make_shared<modl::DBN>(n_input,
// 		std::vector<uint8_t>(n_hiddens.begin(), n_hiddens.end()), label);
// }

eqns::ApproxF get_sgd (PybindT learning_rate)
{
	return [=](const eqns::VarErrsT& leaves)
	{
		return eqns::sgd(leaves, learning_rate);
	};
}

eqns::ApproxF get_rms_momentum (PybindT learning_rate,
	PybindT discount_factor, PybindT epsilon)
{
	return [=](const eqns::VarErrsT& leaves)
	{
		return eqns::rms_momentum(leaves, learning_rate,
			discount_factor, epsilon);
	};
}

}

PYBIND11_MODULE(rocnnet, m)
{
	m.doc() = "rocnnet api";

	py::class_<ade::Shape> shape(m, "Shape");

	// layers
	py::class_<modl::iLayer,modl::LayerptrT> layer(m, "Layer");
	py::class_<modl::Activation,modl::ActivationptrT,modl::iLayer> activation(m, "Activation");
	py::class_<modl::Dense,modl::DenseptrT,modl::iLayer> dense(m, "Dense");
	py::class_<modl::RBM,modl::RBMptrT,modl::iLayer> rbm(m, "RBM");
	py::class_<modl::SequentialModel,modl::SeqModelptrT,modl::iLayer> seqmodel(m, "SequentialModel");

	// trainers
	py::class_<trainer::MLPTrainer> mlptrainer(m, "MLPTrainer");
	py::class_<trainer::DQNTrainer> dqntrainer(m, "DQNTrainer");
	// py::class_<trainer::RBMTrainer> rbmtrainer(m, "RBMTrainer");

	// supports
	py::class_<eqns::VarAssign> assigns(m, "VarAssign");
	py::class_<trainer::DQNInfo> dqninfo(m, "DQNInfo");
	py::class_<trainer::TrainingContext> trainingctx(m, "TrainingContext");
	py::class_<trainer::DQNTrainingContext> dqntrainingctx(m, "DQNTrainingContext");

	shape.def(py::init<std::vector<ade::DimT>>());

	// layer
	layer
		.def("connect", &modl::iLayer::connect)
		.def("get_contents",
			[](py::object self) -> ead::NodesT<PybindT>
			{
				ade::TensT contents = self.cast<modl::iLayer*>()->get_contents();
				ead::NodesT<PybindT> nodes;
				nodes.reserve(contents.size());
				std::transform(contents.begin(), contents.end(),
					std::back_inserter(nodes),
					ead::NodeConverters<PybindT>::to_node);
				return nodes;
			})
		.def("save_file",
			[](py::object self, std::string filename) -> bool
			{
				modl::iLayer& me = *self.cast<modl::iLayer*>();
				std::fstream output(filename,
					std::ios::out | std::ios::trunc | std::ios::binary);
				if (false == modl::save_layer(output, me, me.get_contents()))
				{
					logs::errorf("cannot save to file %s", filename.c_str());
					return false;
				}
				return true;
			})
		.def("save_string",
			[](py::object self) -> std::string
			{
				modl::iLayer& me = *self.cast<modl::iLayer*>();
				std::stringstream savestr;
				modl::save_layer(savestr, me, me.get_contents());
				return savestr.str();
			});

	// activation
	activation
		.def(py::init<const std::string&,const std::string&>(),
			py::arg("label"),
			py::arg("activation_type") = "sigmoid")
		.def("clone", &modl::Activation::clone, py::arg("prefix") = "");

	// dense
	m.def("create_dense",
		[](ead::NodeptrT<PybindT> weight,
			ead::NodeptrT<PybindT> bias,
			std::string label)
		{
			return std::make_shared<modl::Dense>(weight, bias, label);
		},
		py::arg("weight"),
		py::arg("bias") = nullptr,
		py::arg("label"));
	dense
		.def(py::init<ade::DimT,ade::DimT,
			eqns::InitF<PybindT>,
			eqns::InitF<PybindT>,
			const std::string&>(),
			py::arg("nunits"),
			py::arg("indim"),
			py::arg("weight_init") = eqns::unif_xavier_init<PybindT>(1),
			py::arg("bias_init") = eqns::zero_init<PybindT>(),
			py::arg("label"))
		.def("clone", &modl::Dense::clone, py::arg("prefix") = "");

	// rbm
	m.def("create_rbm",
		[](ead::NodeptrT<PybindT> weight,
			ead::NodeptrT<PybindT> hbias,
			ead::NodeptrT<PybindT> vbias,
			std::string label)
		{
			return std::make_shared<modl::RBM>(weight, hbias, vbias, label);
		},
		py::arg("weight"),
		py::arg("hidden_bias") = nullptr,
		py::arg("visible_bias") = nullptr,
		py::arg("label"));
	rbm
		.def(py::init<ade::DimT,ade::DimT,
			eqns::InitF<PybindT>,
			eqns::InitF<PybindT>,
			const std::string&>(),
			py::arg("nhidden"),
			py::arg("nvisible"),
			py::arg("weight_init") = eqns::unif_xavier_init<PybindT>(1),
			py::arg("bias_init") = eqns::zero_init<PybindT>(),
			py::arg("label"))
		.def("clone", &modl::RBM::clone, py::arg("prefix") = "")
		.def("backward_connect", &modl::RBM::backward_connect);

	// seqmodel
	seqmodel
		.def(py::init<const std::string&>(),
			py::arg("label"))
		.def("clone", &modl::SequentialModel::clone, py::arg("prefix") = "")
		.def("add", &modl::SequentialModel::push_back);

	// // dbn
	// m.def("get_dbn", &pyrocnnet::dbn_init);
	// dbn
	// 	.def("copy", [](py::object self)
	// 	{
	// 		return std::make_shared<modl::DBN>(*self.cast<modl::DBN*>());
	// 	}, "deep copy this instance")
	// 	.def("forward", [](py::object self, ead::NodeptrT<PybindT> input)
	// 	{
	// 		return (*self.cast<modl::DBN*>())(input);
	// 	}, "forward input tensor and returned connected output");

	// mlptrainer
	mlptrainer
		.def(py::init<modl::SequentialModel&,
			ead::iSession&,eqns::ApproxF,uint8_t,
			eqns::NodeUnarF,trainer::TrainingContext>(),
			py::arg("model"), py::arg("sess"),
			py::arg("update"), py::arg("batch_size"),
			py::arg("gradprocess") = eqns::NodeUnarF(eqns::identity),
			py::arg("ctx") = trainer::TrainingContext())
		.def("train", &trainer::MLPTrainer::train, "train internal variables")
		.def("train_in",
			[](py::object self)
			{
				return self.cast<trainer::MLPTrainer*>()->train_in_;
			},
			"get train_in variable")
		.def("expected_out",
			[](py::object self)
			{
				return self.cast<trainer::MLPTrainer*>()->expected_out_;
			},
			"get expected_out variable")
		.def("train_out",
			[](py::object self)
			{
				return self.cast<trainer::MLPTrainer*>()->train_out_;
			},
			"get training node")
		.def("error",
			[](py::object self)
			{
				return self.cast<trainer::MLPTrainer*>()->error_;
			},
			"get error node");

	// dqntrainer
	dqninfo
		.def(py::init<size_t,
			PybindT, PybindT, PybindT, PybindT,
			size_t, uint8_t, size_t>(),
			py::arg("train_interval") = 5,
			py::arg("rand_action_prob") = 0.05,
			py::arg("discount_rate") = 0.95,
			py::arg("target_update_rate") = 0.01,
			py::arg("exploration_period") = 1000,
			py::arg("store_interval") = 5,
			py::arg("mini_batch_size") = 32,
			py::arg("max_exp") = 30000);
	dqntrainer
		.def(py::init<modl::SequentialModel&,ead::iSession&,
			eqns::ApproxF,trainer::DQNInfo,
			eqns::NodeUnarF,trainer::DQNTrainingContext>(),
			py::arg("model"), py::arg("sess"),
			py::arg("update"), py::arg("param"),
			py::arg("gradprocess") = eqns::NodeUnarF(eqns::identity),
			py::arg("ctx") = trainer::DQNTrainingContext())
		.def("action", &trainer::DQNTrainer::action, "get next action")
		.def("store", &trainer::DQNTrainer::store, "save observation, action, and reward")
		.def("train", &trainer::DQNTrainer::train, "train qnets")
		.def("error", &trainer::DQNTrainer::get_error, "get prediction error")
		.def("ntrained", &trainer::DQNTrainer::get_numtrained, "get number of iterations trained")
		.def("train_out", [](py::object self)
		{
			return self.cast<trainer::DQNTrainer*>()->train_out_;
		}, "get training node");

	// // rbmtrainer
	// rbmtrainer
	// 	.def(py::init<
	// 		modl::RBMptrT,
	// 		modl::NonLinearsT,
	// 		ead::iSession&,
	// 		ead::VarptrT<PybindT>,
	// 		uint8_t,
	// 		PybindT,
	// 		size_t,
	// 		ead::NodeptrT<PybindT>>(),
	// 		py::arg("brain"),
	// 		py::arg("nolins"),
	// 		py::arg("sess"),
	// 		py::arg("persistent"),
	// 		py::arg("batch_size"),
	// 		py::arg("learning_rate") = 1e-3,
	// 		py::arg("n_cont_div") = 1,
	// 		py::arg("train_in") = nullptr)
	// 	.def("train", &trainer::RBMTrainer::train, "train internal variables")
	// 	.def("train_in", [](py::object self)
	// 	{
	// 		return self.cast<trainer::RBMTrainer*>()->train_in_;
	// 	}, "get train_in variable")
	// 	.def("cost", [](py::object self)
	// 	{
	// 		return self.cast<trainer::RBMTrainer*>()->cost_;
	// 	}, "get cost node")
	// 	.def("monitoring_cost", [](py::object self)
	// 	{
	// 		return self.cast<trainer::RBMTrainer*>()->monitoring_cost_;
	// 	}, "get monitoring cost node")
	// 	.def("brain", [](py::object self)
	// 	{
	// 		return self.cast<trainer::RBMTrainer*>()->brain_;
	// 	}, "get rbm");

	// inlines
	m
		// activations (no longer useful)
		.def("identity", &eqns::identity)

		// optimizations
		.def("get_sgd", &pyrocnnet::get_sgd,
			py::arg("learning_rate") = 0.5)
		.def("get_rms_momentum", &pyrocnnet::get_rms_momentum,
			py::arg("learning_rate") = 0.5,
			py::arg("discount_factor") = 0.99,
			py::arg("epsilon") = std::numeric_limits<PybindT>::epsilon())

		// inits
		.def("variable_from_init",
			[](eqns::InitF<PybindT> init, std::vector<py::ssize_t> slist, std::string label)
			{
				return init(pyrocnnet::p2cshape(slist), label);
			},
			"Return labelled variable containing data created from initializer",
			py::arg("init"), py::arg("slist"), py::arg("label") = "")
		.def("zero_init", eqns::zero_init<PybindT>)
		.def("variance_scaling_init",
			[](PybindT factor)
			{
				return eqns::variance_scaling_init<PybindT>(factor);
			},
			"truncated_normal(shape, 0, sqrt(factor / ((fanin + fanout)/2))",
			py::arg("factor"))
		.def("unif_xavier_init", &eqns::unif_xavier_init<PybindT>,
			"uniform xavier initializer",
			py::arg("factor") = 1)
		.def("norm_xavier_init", &eqns::norm_xavier_init<PybindT>,
			"normal xavier initializer",
			py::arg("factor") = 1)

		// layer creation
		.def("sigmoid", modl::sigmoid, py::arg("label") = "sigmoid")
		.def("tanh", modl::tanh, py::arg("label") = "tanh")
		.def("load_file_seqmodel",
			[](std::string filename, std::string layer_label) -> modl::SeqModelptrT
			{
				std::ifstream input(filename);
				if (false == input.is_open())
				{
					logs::fatalf("file %s not found", filename.c_str());
				}
				ade::TensT trained_roots;
				return std::static_pointer_cast<modl::SequentialModel>(
					modl::load_layer(input, trained_roots, modl::seq_model_key, layer_label));
			})
		.def("load_str_seqmodel",
			[](std::string str, std::string layer_label) -> modl::LayerptrT
			{
				std::istringstream input(str);
				ade::TensT trained_roots;
				return std::static_pointer_cast<modl::SequentialModel>(
					modl::load_layer(input, trained_roots, modl::seq_model_key, layer_label));
			});
};
