#include <fstream>
#include <sstream>

#include "pybind11/stl.h"
#include "pybind11/functional.h"

#include "pyutils/convert.hpp"

#include "layr/trainer/sgd.hpp"
#include "layr/trainer/dqn.hpp"
#include "layr/trainer/dbn.hpp"

namespace py = pybind11;

PYBIND11_MODULE(layr, m)
{
	m.doc() = "layer api";

	// === supports ===
	py::class_<layr::VarAssign<PybindT>> assigns(m, "VarAssign");
	assigns
		.def(py::init(
			[](eteq::EVariable<PybindT> target, eteq::ETensor<PybindT> source)
			{
				return layr::VarAssign<PybindT>{target, source};
			}))
		.def("target",
			[](layr::VarAssign<PybindT>& assign) -> eteq::EVariable<PybindT>
			{ return assign.target_; })
		.def("source",
			[](layr::VarAssign<PybindT>& assign) { return assign.source_; });

	py::class_<trainer::DQNInfo<PybindT>> dqninfo(m, "DQNInfo");
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

	// ==== layer ====
	py::class_<layr::RBMLayer<PybindT>> rbmlayer(m, "RBMLayer");
	rbmlayer
		.def(py::init(
			[](eteq::ELayer<PybindT> fwd, eteq::ELayer<PybindT> bwd)
			{
				return layr::RBMLayer<PybindT>{fwd, bwd};
			}))
		.def("deep_clone", &layr::RBMLayer<PybindT>::deep_clone)
		.def("fwd",
			[](layr::RBMLayer<PybindT>& self)
			{
				return self.fwd_;
			})
		.def("bwd",
			[](layr::RBMLayer<PybindT>& self)
			{
				return self.bwd_;
			})
		.def("connect",
			[](layr::RBMLayer<PybindT>& self, const eteq::ETensor<PybindT>& e)
			{
				return self.fwd_.connect(e);
			})
		.def("backward_connect",
			[](layr::RBMLayer<PybindT>& self, const eteq::ETensor<PybindT>& e)
			{
				return self.bwd_.connect(e);
			});

	// ==== DQN trainer ====
	py::class_<trainer::DQNTrainer<PybindT>> dqntrainer(m, "DQNTrainer");
	dqntrainer
		.def(py::init<eteq::ELayer<PybindT>&,teq::iSession&,
			layr::ApproxF<PybindT>,trainer::DQNInfo<PybindT>,
			layr::UnaryF<PybindT>>(),
			py::arg("model"), py::arg("sess"),
			py::arg("update"), py::arg("param"),
			py::arg("gradprocess") = layr::UnaryF<PybindT>())
		.def("action",
			[](trainer::DQNTrainer<PybindT>* self, py::array input)
			{
				teq::ShapedArr<PybindT> a;
				pyutils::arr2shapedarr(a, input);
				return self->action(a);
			},
			"get next action")
		.def("store", &trainer::DQNTrainer<PybindT>::store, "save observation, action, and reward")
		.def("train", &trainer::DQNTrainer<PybindT>::train, "train qnets")
		.def("error", &trainer::DQNTrainer<PybindT>::get_error, "get prediction error")
		.def("ntrained", &trainer::DQNTrainer<PybindT>::get_numtrained, "get number of iterations trained")
		.def("train_out",
			[](trainer::DQNTrainer<PybindT>* self)
			{
				return self->train_out_;
			}, "get training node");

	// ==== DBN trainer ====
	py::class_<trainer::DBNTrainer<PybindT>> dbntrainer(m, "DBNTrainer");
	dbntrainer
		.def(py::init<
			const std::vector<layr::RBMLayer<PybindT>>&,eteq::ELayer<PybindT>,
			teq::RankT,teq::DimT,PybindT,PybindT,size_t,PybindT,PybindT>(),
			py::arg("rbms"), py::arg("dense"), py::arg("softmax_dim"),
			py::arg("batch_size"), py::arg("pretrain_lr") = 0.1,
			py::arg("train_lr") = 0.1, py::arg("cdk") = 10,
			py::arg("l2_reg") = 0., py::arg("lr_scaling") = 0.95)
		.def("pretrain",
			[](trainer::DBNTrainer<PybindT>* self, py::array x, size_t nepochs,
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
			[](trainer::DBNTrainer<PybindT>* self, py::array x, py::array y, size_t nepochs,
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
		.def("reconstruction_cost", &trainer::DBNTrainer<PybindT>::reconstruction_cost)
		.def("training_cost", &trainer::DBNTrainer<PybindT>::training_cost);

	m
		// ==== inits ====
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

		// ==== layer creation ====
		.def("dense",
			[](std::vector<teq::DimT> inslist, std::vector<teq::DimT> hidden_dims,
				layr::InitF<PybindT> weight_init, layr::InitF<PybindT> bias_init,
				eigen::PairVecT<teq::RankT> dims)
			{
				return layr::dense<PybindT>(teq::Shape(inslist), hidden_dims,
					weight_init, bias_init, dims);
			},
			py::arg("inshape"), py::arg("hidden_dims"),
			py::arg("weight_init") = layr::unif_xavier_init<PybindT>(1),
			py::arg("bias_init") = layr::zero_init<PybindT>(),
			py::arg("dims") = eigen::PairVecT<teq::RankT>{{0, 1}})
		.def("conv", &layr::conv<PybindT>,
			py::arg("filter_hw"), py::arg("in_ncol"), py::arg("out_ncol"),
			py::arg("weight_init") = layr::unif_xavier_init<PybindT>(1),
			py::arg("bias_init") = layr::zero_init<PybindT>(),
			py::arg("zero_padding") = std::pair<teq::DimT,teq::DimT>{0, 0})
		.def("rnn", &layr::rnn<PybindT>,
			py::arg("indim"), py::arg("hidden_dim"), py::arg("activation"), py::arg("nseq"),
			py::arg("weight_init") = layr::unif_xavier_init<PybindT>(1),
			py::arg("bias_init") = layr::zero_init<PybindT>(),
			py::arg("seq_dim") = 1)
		.def("lstm", &layr::lstm<PybindT>,
			py::arg("indim"), py::arg("hidden_dim"), py::arg("nseq"),
			py::arg("weight_init") = layr::unif_xavier_init<PybindT>(1),
			py::arg("bias_init") = layr::zero_init<PybindT>(),
			py::arg("seq_dim") = 1)
		.def("gru", &layr::gru<PybindT>,
			py::arg("indim"), py::arg("hidden_dim"), py::arg("nseq"),
			py::arg("weight_init") = layr::unif_xavier_init<PybindT>(1),
			py::arg("bias_init") = layr::zero_init<PybindT>(),
			py::arg("seq_dim") = 1)
		.def("rbm", &layr::rbm<PybindT>,
			py::arg("nvisible"), py::arg("nhidden"),
			py::arg("weight_init") = layr::unif_xavier_init<PybindT>(1),
			py::arg("bias_init") = layr::zero_init<PybindT>())

		.def("bind", &layr::bind<PybindT>,
			py::arg("unary"), py::arg("inshape") = teq::Shape())
		.def("link",
			[](const std::vector<eteq::ELayer<PybindT>>& layers)
			{
				return layr::link<PybindT>(layers);
			})
		.def("link",
			[](const std::vector<eteq::ELayer<PybindT>>& layers,
				const eteq::ETensor<PybindT>& input)
			{
				return layr::link<PybindT>(layers, input);
			})

		// ==== error functions ====
		.def("sqr_diff", &layr::sqr_diff<PybindT>)

		// ==== layer training ====
		.def("sgd_train", &trainer::sgd<PybindT>,
			py::arg("model"), py::arg("sess"),
			py::arg("train_in"), py::arg("expect_out"), py::arg("update"),
			py::arg("err_func") = layr::ErrorF<PybindT>(layr::sqr_diff<PybindT>),
			py::arg("proc_grad") = layr::UnaryF<PybindT>())
		.def("rbm_train", &trainer::rbm<PybindT>,
			py::arg("rbm_model"), py::arg("sess"), py::arg("visible"),
			py::arg("learning_rate"), py::arg("discount_factor"),
			py::arg("err_func") = layr::ErrorF<PybindT>(),
			py::arg("cdk") = 1)

		// ==== optimizations ====
		.def("get_sgd",
			[](PybindT learning_rate) -> layr::ApproxF<PybindT>
			{
				return [=](const layr::VarErrsT<PybindT>& leaves)
				{
					return layr::sgd(leaves, learning_rate);
				};
			},
			py::arg("learning_rate") = 0.5)
		.def("get_adagrad",
			[](PybindT learning_rate, PybindT epsilon) -> layr::ApproxF<PybindT>
			{
				return [=](const layr::VarErrsT<PybindT>& leaves)
				{
					return layr::adagrad<PybindT>(leaves, learning_rate, epsilon);
				};
			},
			py::arg("learning_rate") = 0.5,
			py::arg("epsilon") = std::numeric_limits<PybindT>::epsilon())
		.def("get_rms_momentum",
			[](PybindT learning_rate, PybindT discount_factor, PybindT epsilon) -> layr::ApproxF<PybindT>
			{
				return [=](const layr::VarErrsT<PybindT>& leaves)
				{
					return layr::rms_momentum<PybindT>(leaves, learning_rate,
						discount_factor, epsilon);
				};
			},
			py::arg("learning_rate") = 0.5,
			py::arg("discount_factor") = 0.99,
			py::arg("epsilon") = std::numeric_limits<PybindT>::epsilon())

		// ==== serialization ====
		.def("load_layers_file",
			[](std::string filename)
			{
				std::ifstream input(filename);
				if (false == input.is_open())
				{
					teq::fatalf("file %s not found", filename.c_str());
				}
				onnx::ModelProto pb_model;
				if (false == pb_model.ParseFromIstream(&input))
				{
					teq::fatalf("failed to parse onnx from %s",
						filename.c_str());
				}
				auto layers = eteq::load_layers<PybindT>(pb_model);
				input.close();
				return layers;
			})
		.def("save_layers_file",
			[](std::string filename, const eteq::ELayersT<PybindT>& models)
			{
				std::ofstream output(filename);
				if (false == output.is_open())
				{
					teq::fatalf("file %s not found", filename.c_str());
				}
				onnx::ModelProto pb_model;
				eteq::save_layers<PybindT>(pb_model, models);
				return pb_model.SerializeToOstream(&output);
			});
}
