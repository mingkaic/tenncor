#include <fstream>
#include <sstream>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"

#include "ead/generated/pyapi.hpp"

#include "rocnnet/eqns/init.hpp"

#include "rocnnet/modl/mlp.hpp"
#include "rocnnet/modl/rbm.hpp"
// #include "rocnnet/modl/dbn.hpp"
// #include "rocnnet/modl/conv.hpp"

#include "rocnnet/trainer/mlp_trainer.hpp"
#include "rocnnet/trainer/dqn_trainer.hpp"
#include "rocnnet/trainer/rbm_trainer.hpp"
// #include "rocnnet/trainer/dbn_trainer.hpp"

namespace py = pybind11;

namespace pyrocnnet
{

ade::Shape p2cshape (std::vector<py::ssize_t>& pyshape)
{
	return ade::Shape(std::vector<ade::DimT>(
		pyshape.rbegin(), pyshape.rend()));
}

trainer::DQNInfo dqninfo_init (size_t train_interval = 5,
	PybindT rand_action_prob = 0.05,
	PybindT discount_rate = 0.95,
	PybindT target_update_rate = 0.01,
	PybindT exploration_period = 1000,
	size_t store_interval = 5,
	uint8_t mini_batch_size = 32,
	size_t max_exp = 30000)
{
	return trainer::DQNInfo{
		train_interval,
		rand_action_prob,
		discount_rate,
		target_update_rate,
		exploration_period,
		store_interval,
		mini_batch_size,
		max_exp,
	};
}

modl::MLPptrT mlp_init (size_t n_input, std::vector<ade::DimT> nouts,
	std::string label)
{
	return std::make_shared<modl::MLP>(n_input, nouts, label);
}

modl::RBMptrT rbm_init (size_t n_input, std::vector<ade::DimT> nouts,
	std::string label)
{
	return std::make_shared<modl::RBM>(n_input, nouts, label);
}

// modl::DBNptrT dbn_init (size_t n_input, std::vector<size_t> n_hiddens,
// 	std::string label)
// {
// 	return std::make_shared<modl::DBN>(n_input,
// 		std::vector<uint8_t>(n_hiddens.begin(), n_hiddens.end()), label);
// }

eqns::ApproxF get_sgd (PybindT learning_rate, eqns::NodeUnarF gradprocess)
{
	return [=](ead::NodeptrT<PybindT>& root, eqns::VariablesT leaves)
	{
		return eqns::sgd(root, leaves, learning_rate, gradprocess);
	};
}

eqns::ApproxF get_rms_momentum (PybindT learning_rate,
	PybindT discount_factor, PybindT epsilon, eqns::NodeUnarF gradprocess)
{
	return [=](ead::NodeptrT<PybindT>& root, eqns::VariablesT leaves)
	{
		return eqns::rms_momentum(root, leaves, learning_rate,
			discount_factor, epsilon, gradprocess);
	};
}

}

PYBIND11_MODULE(rocnnet, m)
{
	m.doc() = "rocnnet api";

	py::class_<ade::Shape> shape(m, "Shape");

	// common parent
	py::class_<modl::iMarshaler,modl::MarsptrT> marshaler(m, "Marshaler");

	// models
	py::class_<modl::MLP,modl::iMarshaler,modl::MLPptrT> mlp(m, "MLP");
	py::class_<modl::RBM,modl::iMarshaler,modl::RBMptrT> rbm(m, "RBM");
	// py::class_<modl::DBN,modl::iMarshaler,modl::DBNptrT> dbn(m, "DBN");

	// support classes
	py::class_<eqns::VarAssign> assigns(m, "VarAssign");

	py::class_<trainer::DQNInfo> dqninfo(m, "DQNInfo");
	py::class_<trainer::MLPTrainer> mlptrainer(m, "MLPTrainer");
	py::class_<trainer::DQNTrainer> dqntrainer(m, "DQNTrainer");
	py::class_<trainer::RBMTrainer> rbmtrainer(m, "RBMTrainer");

	// marshaler
	marshaler
		.def("serialize_to_file", [](py::object self, ead::NodeptrT<PybindT> source,
			std::string filename)
		{
			std::fstream output(filename,
				std::ios::out | std::ios::trunc | std::ios::binary);
			if (false == modl::save(output, source->get_tensor(),
				self.cast<modl::iMarshaler*>()))
			{
				logs::errorf("cannot save to file %s", filename.c_str());
				return false;
			}
			return true;
		}, "load a version of this instance from a data")
		.def("serialize_to_string", [](py::object self, ead::NodeptrT<PybindT> source)
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
		}, "load a version of this instance from a data")
		.def("get_variables", [](py::object self)
		{
			std::unordered_map<std::string,ead::NodeptrT<PybindT>> out;
			pbm::PathedMapT bases = self.cast<modl::iMarshaler*>()->list_bases();
			for (auto bpair : bases)
			{
				if (auto var = std::dynamic_pointer_cast<
					ead::Variable<PybindT>>(bpair.first))
				{
					std::string key = fmts::join("::",
						bpair.second.begin(), bpair.second.end());
					out.emplace(key,
						std::make_shared<ead::VariableNode<PybindT>>(var));
				}
			}
			return out;
		}, "return variables dict in this marshaler");

	// mlp
	m.def("get_mlp", &pyrocnnet::mlp_init);
	mlp
		.def("copy", [](py::object self)
		{
			return std::make_shared<modl::MLP>(*self.cast<modl::MLP*>());
		}, "deep copy this instance")
		.def("forward", [](py::object self, ead::NodeptrT<PybindT> input,
			modl::NonLinearsT nonlins)
		{
			return (*self.cast<modl::MLP*>())(input, nonlins);
		}, "forward input tensor and returned connected output");

	// rbm
	m.def("get_rbm", &pyrocnnet::rbm_init);
	rbm
		.def("copy", [](py::object self)
		{
			return std::make_shared<modl::RBM>(*self.cast<modl::RBM*>());
		}, "deep copy this instance")
		.def("forward", [](py::object self, ead::NodeptrT<PybindT> input,
			modl::NonLinearsT nonlins)
		{
			return (*self.cast<modl::RBM*>())(input, nonlins);
		}, "forward input tensor and returned connected output")
		.def("backward", [](py::object self, ead::NodeptrT<PybindT> hidden,
			modl::NonLinearsT nonlins)
		{
			return self.cast<modl::RBM*>()->prop_down(hidden, nonlins);
		}, "backward hidden tensor and returned connected output")
		.def("reconstruct_visible", [](py::object self,
			ead::NodeptrT<PybindT> input, modl::NonLinearsT nonlins)
		{
			return trainer::reconstruct_visible(
				*self.cast<modl::RBM*>(), input, nonlins);
		}, "reconstruct input")
		.def("reconstruct_hidden", [](py::object self,
			ead::NodeptrT<PybindT> hidden, modl::NonLinearsT nonlins)
		{
			return trainer::reconstruct_hidden(
				*self.cast<modl::RBM*>(), hidden, nonlins);
		}, "reconstruct output");

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
		.def(py::init<modl::MLPptrT,modl::NonLinearsT,
			ead::iSession&,eqns::ApproxF,uint8_t>())
		.def("train", &trainer::MLPTrainer::train, "train internal variables")
		.def("serialize_to_file", [](py::object self, std::string filename)
		{
			std::fstream output(filename,
				std::ios::out | std::ios::trunc | std::ios::binary);
			if (false == self.cast<trainer::MLPTrainer*>()->save(output))
			{
				logs::errorf("cannot save to file %s", filename.c_str());
				return false;
			}
			return true;
		}, "load a version of this instance from a data")
		.def("serialize_to_string", [](py::object self,
			ead::NodeptrT<PybindT> source) -> std::string
		{
			std::stringstream savestr;
			if (self.cast<trainer::MLPTrainer*>()->save(savestr))
			{
				return savestr.str();
			}
			return "";
		}, "load a version of this instance from a data")
		.def("train_in", [](py::object self)
		{
			return self.cast<trainer::MLPTrainer*>()->train_in_;
		}, "get train_in variable")
		.def("expected_out", [](py::object self)
		{
			return self.cast<trainer::MLPTrainer*>()->expected_out_;
		}, "get expected_out variable")
		.def("train_out", [](py::object self)
		{
			return self.cast<trainer::MLPTrainer*>()->train_out_;
		}, "get training node")
		.def("error", [](py::object self)
		{
			return self.cast<trainer::MLPTrainer*>()->error_;
		}, "get error node")
		.def("brain", [](py::object self)
		{
			return self.cast<trainer::MLPTrainer*>()->brain_;
		}, "get mlp");
	m.def("load_mlptrainer", [](std::string data, modl::NonLinearsT nonlins,
		ead::iSession& sess, eqns::ApproxF update,
		uint8_t batch_size) -> trainer::MLPTrainer
	{
		cortenn::Layer layer;
		if (false == layer.ParseFromString(data))
		{
			logs::fatal("failed to parse string when loading mlptrainer");
		}

		// load graph to target
		const cortenn::Graph& graph = layer.graph();
		pbm::GraphInfo info;
		pbm::load_graph<ead::EADLoader>(info, graph);

		auto pretrained = std::make_shared<modl::MLP>(info, "pretrained");

		if (cortenn::Layer::kItCtx != layer.layer_context_case())
		{
			logs::fatal("missing training context");
		}
		trainer::TrainingContext ctx;
		ctx.unmarshal_layer(layer);
		return trainer::MLPTrainer(pretrained, nonlins,
			sess, update, batch_size, ctx);
	});

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
		.def(py::init<modl::MLPptrT,modl::NonLinearsT,
			ead::iSession&,eqns::ApproxF,trainer::DQNInfo>())
		.def("action", &trainer::DQNTrainer::action, "get next action")
		.def("store", &trainer::DQNTrainer::store, "save observation, action, and reward")
		.def("train", &trainer::DQNTrainer::train, "train qnets")
		.def("serialize_to_file", [](py::object self, std::string filename)
		{
			std::fstream output(filename,
				std::ios::out | std::ios::trunc | std::ios::binary);
			if (false == self.cast<trainer::DQNTrainer*>()->save(output))
			{
				logs::errorf("cannot save to file %s", filename.c_str());
				return false;
			}
			return true;
		}, "load a version of this instance from a data")
		.def("serialize_to_string", [](py::object self,
			ead::NodeptrT<PybindT> source) -> std::string
		{
			std::stringstream savestr;
			if (self.cast<trainer::DQNTrainer*>()->save(savestr))
			{
				return savestr.str();
			}
			return "";
		}, "load a version of this instance from a data")
		.def("error", &trainer::DQNTrainer::get_error, "get prediction error")
		.def("ntrained", &trainer::DQNTrainer::get_numtrained, "get number of iterations trained")
		.def("train_out", [](py::object self)
		{
			return self.cast<trainer::DQNTrainer*>()->train_out_;
		}, "get training node");
	m.def("load_dqntrainer", [](std::string data, modl::NonLinearsT nonlins,
		ead::iSession& sess, eqns::ApproxF update,
		trainer::DQNInfo param) -> trainer::DQNTrainer
	{
		cortenn::Layer layer;
		if (false == layer.ParseFromString(data))
		{
			logs::fatal("failed to parse string when loading mlptrainer");
		}

		// load graph to target
		const cortenn::Graph& graph = layer.graph();
		pbm::GraphInfo info;
		pbm::load_graph<ead::EADLoader>(info, graph);

		auto pretrained = std::make_shared<modl::MLP>(info, "pretrained");

		if (cortenn::Layer::kDqnCtx != layer.layer_context_case())
		{
			logs::fatal("missing training context");
		}
		trainer::DQNTrainingContext ctx;
		ctx.unmarshal_layer(layer);
		return trainer::DQNTrainer(pretrained, nonlins,
			sess, update, param, ctx);
	});

	// rbmtrainer
	rbmtrainer
		.def(py::init<
			modl::RBMptrT,
			modl::NonLinearsT,
			ead::iSession&,
			ead::VarptrT<PybindT>,
			uint8_t,
			PybindT,
			size_t,
			ead::NodeptrT<PybindT>>(),
			py::arg("brain"),
			py::arg("nolins"),
			py::arg("sess"),
			py::arg("persistent"),
			py::arg("batch_size"),
			py::arg("learning_rate") = 1e-3,
			py::arg("n_cont_div") = 1,
			py::arg("train_in") = nullptr)
		.def("train", &trainer::RBMTrainer::train, "train internal variables")
		.def("train_in", [](py::object self)
		{
			return self.cast<trainer::RBMTrainer*>()->train_in_;
		}, "get train_in variable")
		.def("cost", [](py::object self)
		{
			return self.cast<trainer::RBMTrainer*>()->cost_;
		}, "get cost node")
		.def("monitoring_cost", [](py::object self)
		{
			return self.cast<trainer::RBMTrainer*>()->monitoring_cost_;
		}, "get monitoring cost node")
		.def("brain", [](py::object self)
		{
			return self.cast<trainer::RBMTrainer*>()->brain_;
		}, "get rbm");


	// inlines
	m.def("identity", &eqns::identity);

	m.def("get_sgd", &pyrocnnet::get_sgd,
		py::arg("learning_rate") = 0.5,
		py::arg("gradprocess") = eqns::NodeUnarF(eqns::identity));
	m.def("get_rms_momentum", &pyrocnnet::get_rms_momentum,
		py::arg("learning_rate") = 0.5,
		py::arg("discount_factor") = 0.99,
		py::arg("epsilon") = std::numeric_limits<PybindT>::epsilon(),
		py::arg("gradprocess") = eqns::NodeUnarF(eqns::identity));

	m.def("variable_from_init",
	[](eqns::InitF<PybindT> init, std::vector<py::ssize_t> slist, std::string label)
	{
		return init(pyrocnnet::p2cshape(slist), label);
	},
	"Return labelled variable containing data created from initializer",
	py::arg("init"), py::arg("slist"), py::arg("label") = "");

	m.def("variance_scaling_init", [](PybindT factor)
	{
		return eqns::variance_scaling_init<PybindT>(factor);
	},
	"truncated_normal(shape, 0, sqrt(factor / ((fanin + fanout)/2))",
	py::arg("factor"));

	m.def("unif_xavier_init", &eqns::unif_xavier_init<PybindT>,
	"uniform xavier initializer",
	py::arg("factor") = 1);

	m.def("norm_xavier_init", &eqns::norm_xavier_init<PybindT>,
	"normal xavier initializer",
	py::arg("factor") = 1);
};
