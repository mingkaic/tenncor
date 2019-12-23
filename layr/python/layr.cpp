#include <fstream>
#include <sstream>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"

#include "pyutils/convert.hpp"

#include "eteq/generated/pyapi.hpp"

#include "layr/init.hpp"
#include "layr/approx.hpp"
#include "layr/api.hpp"
#include "layr/trainer/sgd.hpp"
#include "layr/trainer/rbm.hpp"

namespace py = pybind11;

PYBIND11_MODULE(layr, m)
{
	m.doc() = "layer api";

	py::class_<layr::RBMLayer<PybindT>> rbmlayer(m, "RBMLayer");

	m
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
			py::arg("weight_init"), py::arg("bias_init"),
			py::arg("dims") = eigen::PairVecT<teq::RankT>{{0, 1}})
		.def("conv", &layr::conv<PybindT>,
			py::arg("filter_hw"), py::arg("in_ncol"), py::arg("out_ncol"),
			py::arg("zero_padding") = std::pair<teq::DimT,teq::DimT>{0, 0})
		.def("rnn", &layr::rnn<PybindT>,
			py::arg("indim"), py::arg("hidden_dim"), py::arg("activation"),
			py::arg("weight_init"), py::arg("bias_init"),
			py::arg("nseq"), py::arg("seq_dim") = 1)
		.def("lstm", &layr::lstm<PybindT>,
			py::arg("indim"), py::arg("hidden_dim"),
			py::arg("weight_init"), py::arg("bias_init"),
			py::arg("nseq"), py::arg("seq_dim") = 1)
		.def("gru", &layr::gru<PybindT>,
			py::arg("indim"), py::arg("hidden_dim"),
			py::arg("weight_init"), py::arg("bias_init"),
			py::arg("nseq"), py::arg("seq_dim") = 1)
		.def("rbm", &layr::rbm<PybindT>,
			py::arg("nhidden"), py::arg("nvisible"),
			py::arg("weight_init"), py::arg("bias_init"))

		// ==== layer training ====
		.def("sgd_train", &trainer::sgd<PybindT>,
			py::arg("model"), py::arg("sess"),
			py::arg("train_in"), py::arg("expect_out"), py::arg("update"),
			py::arg("errfunc") = layr::ErrorF<PybindT>(layr::sqr_diff<PybindT>),
			py::arg("proc_grad") = layr::UnaryF<PybindT>())
		.def("rbm_train", &trainer::rbm<PybindT>,
			py::arg("rbm_model"), py::arg("sess"), py::arg("visible"),
			py::arg("learning_rate"), py::arg("discount_factor"),
			py::arg("errfunc") = layr::ErrorF<PybindT>(),
			py::arg("cdk") = 1);
}
