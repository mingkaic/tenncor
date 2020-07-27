
#include "python/layr_ext.hpp"

void layr_ext(py::module& m)
{
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

	// ==== DBN trainer ====
	py::class_<trainer::DBNTrainer<PybindT>> dbntrainer(m, "DBNTrainer");

	dbntrainer
		.def(py::init<
			const std::vector<layr::RBMLayer<PybindT>>&,eteq::ETensor<PybindT>,
			teq::RankT,teq::DimT,PybindT,PybindT,size_t,PybindT,PybindT,eigen::CtxptrT>(),
			py::arg("rbms"), py::arg("dense"),
			py::arg("softmax_dim"), py::arg("batch_size"),
			py::arg("pretrain_lr") = 0.1,
			py::arg("train_lr") = 0.1, py::arg("cdk") = 10,
			py::arg("l2_reg") = 0., py::arg("lr_scaling") = 0.95,
			py::arg("context") = eigen::global_context())
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

		// ==== layer training ====
		.def("apply_update",
		[](const eteq::ETensorsT<PybindT>& models,
			layr::ApproxF<PybindT> update, layr::ErrorF<PybindT> err_func, eigen::CtxptrT ctx)
		{
			return trainer::apply_update<PybindT>(models, update, err_func, ctx);
		}, py::arg("models"), py::arg("update"), py::arg("err_func"), py::arg("ctx") = eigen::global_context())
		.def("rbm_train", &trainer::rbm<PybindT>,
			py::arg("rbm_model"), py::arg("visible"),
			py::arg("learning_rate"), py::arg("discount_factor"),
			py::arg("err_func") = layr::BErrorF<PybindT>(
			[](const pytenncor::ETensT& l, const pytenncor::ETensT& r)
			{
				return tenncor<PybindT>().error.sqr_diff(l, r);
			}),
			py::arg("cdk") = 1,
			py::arg("ctx") = eigen::global_context());
}
