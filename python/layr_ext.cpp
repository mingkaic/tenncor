#include "python/layr_ext.hpp"

namespace pylayr
{

layr::VarMapT<PybindT> convert (const VPairsT& op)
{
	layr::VarMapT<PybindT> out;
	for (auto& o : op)
	{
		out.emplace(o.first, o.second);
	}
	return out;
}

VPairsT convert (const layr::VarMapT<PybindT>& op)
{
	VPairsT out;
	for (auto& o : op)
	{
		out.push_back({eteq::EVariable<PybindT>(o.first), o.second});
	}
	return out;
}

layr::ApproxF<PybindT> convert (ApproxF f)
{
	return [f](const layr::VarMapT<PybindT>& vs)
	{
		return convert(f(convert(vs)));
	};
}

}

void layr_ext(py::module& m)
{
	m.attr("bind_name") = layr::bind_name;
	m.attr("link_name") = layr::link_name;
	m.attr("dense_name") = layr::dense_name;
	m.attr("conv_name") = layr::conv_name;
	m.attr("rnn_name") = layr::rnn_name;
	m.attr("lstm_name") = layr::lstm_name;
	m.attr("gru_name") = layr::gru_name;

	// === supports ===
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
			[](eteq::ETensor<PybindT> fwd, eteq::ETensor<PybindT> bwd)
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
		.def("connect", &layr::RBMLayer<PybindT>::connect)
		.def("backward_connect", &layr::RBMLayer<PybindT>::backward_connect);

	// ==== DQN trainer ====
	py::class_<trainer::DQNTrainer<PybindT>> dqntrainer(m, "DQNTrainer");
	dqntrainer
		.def(py::init([](const std::string& layername,
				eteq::ETensor<PybindT>& model, teq::iSession& sess,
				layr::ApproxF<PybindT> update, trainer::DQNInfo<PybindT> param,
				layr::UnaryF<PybindT> gradprocess)
			{
				return trainer::DQNTrainer<PybindT>(layername, model, sess,
					update, param, gradprocess);
			}),
			py::arg("layername"), py::arg("model"), py::arg("sess"),
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
			const std::vector<layr::RBMLayer<PybindT>>&,eteq::ETensor<PybindT>,
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
			[](layr::InitF<PybindT> init, py::list slist, std::string label)
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
		.def("bind", &layr::bind<PybindT>,
			py::arg("unary"), py::arg("inshape") = teq::Shape())
		.def("link",
			[](const layr::LayersT<PybindT>& layers)
			{
				return layr::link<PybindT>(layers);
			})
		.def("link",
			[](const layr::LayersT<PybindT>& layers,
				const eteq::ETensor<PybindT>& input)
			{
				return layr::link<PybindT>(layers, input);
			})

		// ==== layer training ====
		.def("sgd_train", [](const std::string& layername,
				const eteq::ETensor<PybindT>& model, eteq::ETensor<PybindT> train_in,
				eteq::ETensor<PybindT> expect_out, layr::ApproxF<PybindT> update,
				layr::ErrorF<PybindT> err_func, layr::UnaryF<PybindT> proc_grad)
			{
				return trainer::sgd<PybindT>(layername, model, train_in, expect_out, update, err_func, proc_grad);
			},
			py::arg("layername"), py::arg("model"), py::arg("train_in"),
			py::arg("expect_out"), py::arg("update"),
			py::arg("err_func") = layr::ErrorF<PybindT>(tenncor::error::sqr_diff<PybindT>),
			py::arg("proc_grad") = layr::UnaryF<PybindT>())
		.def("rbm_train", &trainer::rbm<PybindT>,
			py::arg("rbm_model"), py::arg("visible"),
			py::arg("learning_rate"), py::arg("discount_factor"),
			py::arg("err_func") = layr::ErrorF<PybindT>(tenncor::error::sqr_diff<PybindT>),
			py::arg("cdk") = 1);
}
