#include <fstream>
#include <sstream>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"

#include "pyutils/convert.hpp"

#include "eteq/generated/pyapi.hpp"

#include "layr/init.hpp"

#include "layr/ulayer.hpp"
#include "layr/dense.hpp"
#include "layr/rbm.hpp"
#include "layr/conv.hpp"
#include "layr/rnn.hpp"
#include "layr/lstm.hpp"
#include "layr/gru.hpp"
#include "layr/seqmodel.hpp"

#include "rocnnet/trainer/sgd_trainer.hpp"
#include "rocnnet/trainer/dqn_trainer.hpp"
#include "rocnnet/trainer/rbm_trainer.hpp"
#include "rocnnet/trainer/dbn_trainer.hpp"

namespace py = pybind11;

namespace pyrocnnet
{

layr::ApproxF get_sgd (PybindT learning_rate)
{
	return [=](const layr::VarErrsT& leaves)
	{
		return layr::sgd(leaves, learning_rate);
	};
}

layr::ApproxF get_adagrad (PybindT learning_rate, PybindT epsilon)
{
	return [=](const layr::VarErrsT& leaves)
	{
		return layr::adagrad(leaves, learning_rate, epsilon);
	};
}

layr::ApproxF get_rms_momentum (PybindT learning_rate,
	PybindT discount_factor, PybindT epsilon)
{
	return [=](const layr::VarErrsT& leaves)
	{
		return layr::rms_momentum(leaves, learning_rate,
			discount_factor, epsilon);
	};
}

eteq::NodeptrT<PybindT> identity (eteq::NodeptrT<PybindT> in)
{ return in; }

}

PYBIND11_MODULE(rocnnet, m)
{
	m.doc() = "rocnnet api";

	// layers
	py::class_<layr::iLayer,layr::LayerptrT> layer(m, "Layer");
	py::class_<layr::ULayer,layr::UnaryptrT,layr::iLayer> ulayer(m, "ULayer");
	py::class_<layr::Dense,layr::DenseptrT,layr::iLayer> dense(m, "Dense");
	py::class_<layr::RBM,layr::RBMptrT,layr::iLayer> rbm(m, "RBM");
	py::class_<layr::Conv,layr::ConvptrT,layr::iLayer> conv(m, "Conv");
	py::class_<layr::RNN,layr::RNNptrT,layr::iLayer> rnn(m, "RNN");
	py::class_<layr::LSTM,layr::LSTMptrT,layr::iLayer> lstm(m, "LSTM");
	py::class_<layr::GRU,layr::GRUptrT,layr::iLayer> gru(m, "GRU");
	py::class_<layr::SequentialModel,layr::SeqModelptrT,layr::iLayer> seqmodel(m, "SequentialModel");

	// trainers
	py::class_<trainer::DQNTrainer> dqntrainer(m, "DQNTrainer");
	py::class_<trainer::DBNTrainer> dbntrainer(m, "DBNTrainer");

	// supports
	py::class_<layr::VarAssign> assigns(m, "VarAssign");
	py::class_<trainer::DQNInfo> dqninfo(m, "DQNInfo");
	py::class_<trainer::DQNTrainingContext> dqntrainingctx(m, "DQNTrainingContext");

	assigns
		.def(py::init(
			[](eteq::VarptrT<PybindT> target, NodeptrT source)
			{
				return layr::VarAssign{"", target, source};
			}))
		.def("target", [](layr::VarAssign& assign) { return assign.target_; })
		.def("source", [](layr::VarAssign& assign) { return assign.source_; });

	dqninfo
		.def(py::init<size_t,
			PybindT, PybindT, PybindT, PybindT,
			size_t, teq::DimT, size_t>(),
			py::arg("train_interval") = 5,
			py::arg("rand_action_prob") = 0.05,
			py::arg("discount_rate") = 0.95,
			py::arg("target_update_rate") = 0.01,
			py::arg("exploration_period") = 1000,
			py::arg("store_interval") = 5,
			py::arg("mini_batch_size") = 32,
			py::arg("max_exp") = 30000);

	// layer
	layer
		.def("connect", &layr::iLayer::connect)
		.def("get_contents",
			[](layr::iLayer* self) -> eteq::NodesT<PybindT>
			{
				teq::TensptrsT contents = self->get_contents();
				eteq::NodesT<PybindT> nodes;
				nodes.reserve(contents.size());
				std::transform(contents.begin(), contents.end(),
					std::back_inserter(nodes),
					[](teq::TensptrT tens)
					{ return eteq::to_node<PybindT>(tens); });
				return nodes;
			})
		.def("save_file",
			[](layr::iLayer* self, std::string filename) -> bool
			{
				std::fstream output(filename,
					std::ios::out | std::ios::trunc | std::ios::binary);
				if (false == layr::save_layer(output, *self, self->get_contents()))
				{
					logs::errorf("cannot save to file %s", filename.c_str());
					return false;
				}
				return true;
			})
		.def("save_string",
			[](layr::iLayer* self) -> std::string
			{
				std::stringstream savestr;
				layr::save_layer(savestr, *self, self->get_contents());
				return savestr.str();
			})
		.def("get_ninput", &layr::iLayer::get_ninput)
		.def("get_noutput", &layr::iLayer::get_noutput);

	// ulayer
	ulayer
		.def(py::init<const std::string&,NodeptrT,const std::string&>(),
			py::arg("ulayer_type"),
			py::arg("params"),
			py::arg("label"))
		.def("clone", &layr::ULayer::clone, py::arg("prefix") = "");

	// dense
	m.def("create_dense",
		[](NodeptrT weight,
			NodeptrT bias,
			NodeptrT params,
			std::string label)
		{
			return std::make_shared<layr::Dense>(weight, bias, params, label);
		},
		py::arg("weight"),
		py::arg("bias") = nullptr,
		py::arg("params") = nullptr,
		py::arg("label") = "");
	dense
		.def(py::init<teq::DimT,teq::Shape,
			layr::InitF<PybindT>,
			layr::InitF<PybindT>,
			NodeptrT,
			const std::string&>(),
			py::arg("nunits"),
			py::arg("inshape"),
			py::arg("weight_init") = layr::unif_xavier_init<PybindT>(1),
			py::arg("bias_init") = layr::zero_init<PybindT>(),
			py::arg("params") = nullptr,
			py::arg("label") = "")
		.def("clone", &layr::Dense::clone, py::arg("prefix") = "");

	// rbm
	m.def("create_rbm",
		[](layr::DenseptrT hidden,
			layr::DenseptrT visible,
			layr::UnaryptrT activation,
			std::string label)
		{
			return std::make_shared<layr::RBM>(
				hidden, visible, activation, label);
		},
		py::arg("hidden"),
		py::arg("visible") = nullptr,
		py::arg("activation") = nullptr,
		py::arg("label") = "");
	rbm
		.def(py::init<teq::DimT,teq::DimT,
			layr::UnaryptrT,
			layr::InitF<PybindT>,
			layr::InitF<PybindT>,
			const std::string&>(),
			py::arg("nhidden"),
			py::arg("nvisible"),
			py::arg("activation") = layr::sigmoid(),
			py::arg("weight_init") = layr::unif_xavier_init<PybindT>(1),
			py::arg("bias_init") = layr::zero_init<PybindT>(),
			py::arg("label") = "")
		.def("clone", &layr::RBM::clone, py::arg("prefix") = "")
		.def("backward_connect", &layr::RBM::backward_connect);

	// conv
	m.def("create_conv",
		[](NodeptrT weight, NodeptrT bias,
			eteq::NodeptrT<layr::DArgT> arg,
			std::string label)
		{
			return std::make_shared<layr::Conv>(
				weight, bias, arg, label);
		},
		py::arg("weight"),
		py::arg("bias"),
		py::arg("arg"),
		py::arg("label") = "");
	conv
		.def(py::init<std::pair<teq::DimT,teq::DimT>,
			teq::DimT,teq::DimT,const std::string&,
			std::pair<teq::DimT,teq::DimT>>(),
			py::arg("filter_hw"),
			py::arg("in_ncol"),
			py::arg("out_ncol"),
			py::arg("label") = "",
			py::arg("zero_padding") = std::pair<teq::DimT,teq::DimT>{0, 0})
		.def("clone", &layr::Conv::clone, py::arg("prefix") = "")
		.def("paddings", &layr::Conv::get_padding);

	// rnn
	m.def("create_recur",
		[](layr::DenseptrT cell, layr::UnaryptrT activation,
			NodeptrT init_state, NodeptrT param, std::string label)
		{
			return std::make_shared<layr::RNN>(
				cell, activation, init_state, param, label);
		},
		py::arg("cell"),
		py::arg("activation"),
		py::arg("init_state"),
		py::arg("param"),
		py::arg("label") = "");
	rnn
		.def(py::init<teq::DimT,layr::UnaryptrT,
			layr::InitF<PybindT>,layr::InitF<PybindT>,
			teq::RankT,const std::string&>(),
			py::arg("nunits"), py::arg("activation"),
			py::arg("weight_init"), py::arg("bias_init"),
			py::arg("seq_dim"), py::arg("label") = "")
		.def("clone", &layr::RNN::clone, py::arg("prefix") = "");

	// lstm
	m.def("create_lstm",
		[](layr::DenseptrT gate,
			layr::DenseptrT forget,
			layr::DenseptrT ingate,
			layr::DenseptrT outgate,
			std::string label)
		{
			return std::make_shared<layr::LSTM>(
				gate, forget, ingate, outgate, label);
		},
		py::arg("gate"),
		py::arg("forget"),
		py::arg("ingate"),
		py::arg("outgate"),
		py::arg("label") = "");
	lstm
		.def(py::init<teq::DimT,teq::DimT,
			layr::InitF<PybindT>,
			layr::InitF<PybindT>,
			const std::string&>(),
			py::arg("nhidden"),
			py::arg("ninput"),
			py::arg("weight_init"),
			py::arg("bias_init"),
			py::arg("label") = "")
		.def("clone", &layr::LSTM::clone, py::arg("prefix") = "");

	// gru
	m.def("create_gru",
		[](layr::DenseptrT ugate,
			layr::DenseptrT rgate,
			layr::DenseptrT hgate,
			std::string label)
		{
			return std::make_shared<layr::GRU>(
				ugate, rgate, hgate, label);
		},
		py::arg("ugate"),
		py::arg("rgate"),
		py::arg("hgate"),
		py::arg("label") = "");
	gru
		.def(py::init<teq::DimT,teq::DimT,
			layr::InitF<PybindT>,
			layr::InitF<PybindT>,
			const std::string&>(),
			py::arg("nhidden"),
			py::arg("ninput"),
			py::arg("weight_init"),
			py::arg("bias_init"),
			py::arg("label") = "")
		.def("clone", &layr::GRU::clone, py::arg("prefix") = "");

	// seqmodel
	seqmodel
		.def(py::init<const std::string&>(),
			py::arg("label"))
		.def("clone", &layr::SequentialModel::clone, py::arg("prefix") = "")
		.def("add", &layr::SequentialModel::push_back)
		.def("get_layers", &layr::SequentialModel::get_layers);


	// dqntrainer
	dqntrainer
		.def(py::init<layr::SequentialModel&,teq::iSession&,
			layr::ApproxF,trainer::DQNInfo,
			trainer::NodeUnarF,trainer::DQNTrainingContext>(),
			py::arg("model"), py::arg("sess"),
			py::arg("update"), py::arg("param"),
			py::arg("gradprocess") = trainer::NodeUnarF(pyrocnnet::identity),
			py::arg("ctx") = trainer::DQNTrainingContext())
		.def("action",
			[](trainer::DQNTrainer* self, py::array input)
			{
				teq::ShapedArr<PybindT> a;
				pyutils::arr2shapedarr(a, input);
				return self->action(a);
			},
			"get next action")
		.def("store", &trainer::DQNTrainer::store, "save observation, action, and reward")
		.def("train", &trainer::DQNTrainer::train, "train qnets")
		.def("error", &trainer::DQNTrainer::get_error, "get prediction error")
		.def("ntrained", &trainer::DQNTrainer::get_numtrained, "get number of iterations trained")
		.def("train_out",
			[](trainer::DQNTrainer* self)
			{
				return self->train_out_;
			}, "get training node");

	// dbntrainer
	dbntrainer
		.def(py::init<
			layr::SequentialModel& ,
			uint8_t,PybindT,PybindT,
			size_t,PybindT,PybindT>(),
			py::arg("model"),
			py::arg("batch_size"),
			py::arg("pretrain_lr") = 0.1,
			py::arg("train_lr") = 0.1,
			py::arg("cdk") = 10,
			py::arg("l2_reg") = 0.,
			py::arg("lr_scaling") = 0.95)
		.def("pretrain",
			[](trainer::DBNTrainer* self, py::array x, size_t nepochs,
				std::function<void(size_t,size_t)> logger)
			{
				teq::ShapedArr<PybindT> xa;
				pyutils::arr2shapedarr(xa, x);
				return self->pretrain(xa, nepochs, logger);
			},
			py::arg("x"),
			py::arg("nepochs") = 100,
			py::arg("logger") = std::function<void(size_t,size_t)>(),
			"pretrain internal rbms")
		.def("finetune",
			[](trainer::DBNTrainer* self, py::array x, py::array y, size_t nepochs,
				std::function<void(size_t)> logger)
			{
				teq::Shape shape;
				teq::ShapedArr<PybindT> xa;
				teq::ShapedArr<PybindT> ya;
				pyutils::arr2shapedarr(xa, x);
				pyutils::arr2shapedarr(ya, y);
				return self->finetune(xa, ya, nepochs, logger);
			},
			py::arg("x"),
			py::arg("y"),
			py::arg("nepochs") = 100,
			py::arg("logger") = std::function<void(size_t)>(),
			"finetune internal dense layer")
		.def("reconstruction_cost", &trainer::DBNTrainer::reconstruction_cost)
		.def("training_cost", &trainer::DBNTrainer::training_cost);

	// inlines
	m
		// optimizations
		.def("get_sgd", &pyrocnnet::get_sgd,
			py::arg("learning_rate") = 0.5)
		.def("get_adagrad", &pyrocnnet::get_adagrad,
			py::arg("learning_rate") = 0.5,
			py::arg("epsilon") = std::numeric_limits<PybindT>::epsilon())
		.def("get_rms_momentum", &pyrocnnet::get_rms_momentum,
			py::arg("learning_rate") = 0.5,
			py::arg("discount_factor") = 0.99,
			py::arg("epsilon") = std::numeric_limits<PybindT>::epsilon())

		// inits
		.def("variable_from_init",
			[](layr::InitF<PybindT> init, std::vector<py::ssize_t> slist,
				std::string label)
			{
				return init(pyutils::p2cshape(slist), label);
			},
			"Return labelled variable containing data created from initializer",
			py::arg("init"), py::arg("slist"), py::arg("label") = "")
		.def("zero_init", layr::zero_init<PybindT>)
		.def("variance_scaling_init",
			[](PybindT factor)
			{
				return layr::variance_scaling_init<PybindT>(factor);
			},
			"truncated_normal(shape, 0, sqrt(factor / ((fanin + fanout)/2))",
			py::arg("factor"))
		.def("unif_xavier_init", &layr::unif_xavier_init<PybindT>,
			"uniform xavier initializer",
			py::arg("factor") = 1)
		.def("norm_xavier_init", &layr::norm_xavier_init<PybindT>,
			"normal xavier initializer",
			py::arg("factor") = 1)

		// layer creation
		.def("sigmoid", layr::sigmoid, py::arg("label") = "")
		.def("tanh", layr::tanh, py::arg("label") = "")
		.def("relu", layr::relu, py::arg("label") = "")
		.def("softmax", layr::softmax,
			py::arg("dim"),
			py::arg("label") = "")
		.def("maxpool2d", layr::maxpool2d,
			py::arg("dims") = std::pair<teq::DimT,teq::DimT>{0, 1},
			py::arg("label") = "")
		.def("meanpool2d", layr::meanpool2d,
			py::arg("dims") = std::pair<teq::DimT,teq::DimT>{0, 1},
			py::arg("label") = "")
		.def("load_file_seqmodel",
			[](std::string filename, std::string layer_label) -> layr::SeqModelptrT
			{
				std::ifstream input(filename);
				if (false == input.is_open())
				{
					logs::fatalf("file %s not found", filename.c_str());
				}
				teq::TensptrsT trained_roots;
				return std::static_pointer_cast<layr::SequentialModel>(
					layr::load_layer(input, trained_roots, layr::seq_model_key, layer_label));
			})
		.def("load_file_rbmmodel",
			[](std::string filename, std::string layer_label) -> layr::RBMptrT
			{
				std::ifstream input(filename);
				if (false == input.is_open())
				{
					logs::fatalf("file %s not found", filename.c_str());
				}
				teq::TensptrsT trained_roots;
				return std::static_pointer_cast<layr::RBM>(
					layr::load_layer(input, trained_roots, layr::rbm_layer_key, layer_label));
			})

		// trainers
		.def("sgd_train", &trainer::sgd_train,
			py::arg("model"), py::arg("sess"),
			py::arg("train_in"), py::arg("expected_out"), py::arg("update"),
			py::arg("errfunc") = layr::ErrorF(layr::sqr_diff),
			py::arg("gradprocess") = trainer::NodeUnarF(pyrocnnet::identity))
		.def("rbm_train", &trainer::rbm_train,
			py::arg("model"), py::arg("sess"),
			py::arg("visible"), py::arg("learning_rate"),
			py::arg("discount_factor"), py::arg("err_func"),
			py::arg("cdk") = 1);
};
