#include <fstream>
#include <sstream>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"

#include "eteq/generated/pyapi.hpp"

#include "layr/init.hpp"

#include "layr/ulayer.hpp"
#include "layr/dense.hpp"
#include "layr/rbm.hpp"
#include "layr/seqmodel.hpp"
#include "layr/conv.hpp"

#include "rocnnet/trainer/mlp_trainer.hpp"
#include "rocnnet/trainer/dqn_trainer.hpp"
#include "rocnnet/trainer/rbm_trainer.hpp"
#include "rocnnet/trainer/dbn_trainer.hpp"

namespace py = pybind11;

namespace pyrocnnet
{

teq::Shape p2cshape (std::vector<py::ssize_t>& pyshape)
{
	return teq::Shape(std::vector<teq::DimT>(
		pyshape.rbegin(), pyshape.rend()));
}

std::vector<PybindT> arr2vec (teq::Shape& outshape, py::array data)
{
	py::buffer_info info = data.request();
	outshape = p2cshape(info.shape);
	size_t n = outshape.n_elems();
	auto dtype = data.dtype();
	char kind = dtype.kind();
	py::ssize_t tbytes = dtype.itemsize();
	std::vector<PybindT> vec;
	switch (kind)
	{
		case 'f':
			switch (tbytes)
			{
				case 4: // float32
				{
					float* dptr = static_cast<float*>(info.ptr);
					vec = std::vector<PybindT>(dptr, dptr + n);
				}
				break;
				case 8: // float64
				{
					double* dptr = static_cast<double*>(info.ptr);
					vec = std::vector<PybindT>(dptr, dptr + n);
				}
					break;
				default:
					logs::fatalf("unsupported float type with %d bytes", tbytes);
			}
			break;
		case 'i':
			switch (tbytes)
			{
				case 1: // int8
				{
					int8_t* dptr = static_cast<int8_t*>(info.ptr);
					vec = std::vector<PybindT>(dptr, dptr + n);
				}
					break;
				case 2: // int16
				{
					int16_t* dptr = static_cast<int16_t*>(info.ptr);
					vec = std::vector<PybindT>(dptr, dptr + n);
				}
					break;
				case 4: // int32
				{
					int32_t* dptr = static_cast<int32_t*>(info.ptr);
					vec = std::vector<PybindT>(dptr, dptr + n);
				}
					break;
				case 8: // int64
				{
					int64_t* dptr = static_cast<int64_t*>(info.ptr);
					vec = std::vector<PybindT>(dptr, dptr + n);
				}
					break;
				default:
					logs::fatalf("unsupported integer type with %d bytes", tbytes);
			}
			break;
		default:
			logs::fatalf("unknown dtype %c", kind);
	}
	return vec;
}

layr::ApproxF get_sgd (PybindT learning_rate)
{
	return [=](const layr::VarErrsT& leaves)
	{
		return layr::sgd(leaves, learning_rate);
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

	py::class_<teq::Shape> shape(m, "Shape");

	// layers
	py::class_<layr::iLayer,layr::LayerptrT> layer(m, "Layer");
	py::class_<layr::ULayer,layr::UnaryptrT,layr::iLayer> ulayer(m, "ULayer");
	py::class_<layr::Dense,layr::DenseptrT,layr::iLayer> dense(m, "Dense");
	py::class_<layr::RBM,layr::RBMptrT,layr::iLayer> rbm(m, "RBM");
	py::class_<layr::Conv,layr::ConvptrT,layr::iLayer> conv(m, "Conv");
	py::class_<layr::SequentialModel,layr::SeqModelptrT,layr::iLayer> seqmodel(m, "SequentialModel");

	// trainers
	py::class_<trainer::MLPTrainer> mlptrainer(m, "MLPTrainer");
	py::class_<trainer::DQNTrainer> dqntrainer(m, "DQNTrainer");
	py::class_<trainer::BernoulliRBMTrainer> brbmtrainer(m, "BernoulliRBMTrainer");
	py::class_<trainer::DBNTrainer> dbntrainer(m, "DBNTrainer");

	// supports
	py::class_<layr::VarAssign> assigns(m, "VarAssign");
	py::class_<trainer::DQNInfo> dqninfo(m, "DQNInfo");
	py::class_<trainer::TrainingContext> trainingctx(m, "TrainingContext");
	py::class_<trainer::DQNTrainingContext> dqntrainingctx(m, "DQNTrainingContext");

	shape
		.def(py::init<std::vector<teq::DimT>>())
		.def("__getitem__",
			[](teq::Shape& shape, size_t idx) { return shape.at(idx); },
			py::is_operator());

	// layer
	layer
		.def("connect", &layr::iLayer::connect)
		.def("get_contents",
			[](py::object self) -> eteq::NodesT<PybindT>
			{
				teq::TensptrsT contents = self.cast<layr::iLayer*>()->get_contents();
				eteq::NodesT<PybindT> nodes;
				nodes.reserve(contents.size());
				std::transform(contents.begin(), contents.end(),
					std::back_inserter(nodes),
					eteq::NodeConverters<PybindT>::to_node);
				return nodes;
			})
		.def("save_file",
			[](py::object self, std::string filename) -> bool
			{
				layr::iLayer& me = *self.cast<layr::iLayer*>();
				std::fstream output(filename,
					std::ios::out | std::ios::trunc | std::ios::binary);
				if (false == layr::save_layer(output, me, me.get_contents()))
				{
					logs::errorf("cannot save to file %s", filename.c_str());
					return false;
				}
				return true;
			})
		.def("save_string",
			[](py::object self) -> std::string
			{
				layr::iLayer& me = *self.cast<layr::iLayer*>();
				std::stringstream savestr;
				layr::save_layer(savestr, me, me.get_contents());
				return savestr.str();
			})
		.def("get_ninput", &layr::iLayer::get_ninput)
		.def("get_noutput", &layr::iLayer::get_noutput);

	// ulayer
	ulayer
		.def(py::init<const std::string&,const std::string&>(),
			py::arg("label"),
			py::arg("ulayer_type") = "sigmoid")
		.def("clone", &layr::ULayer::clone, py::arg("prefix") = "");

	// dense
	m.def("create_dense",
		[](NodeptrT weight,
			NodeptrT bias,
			std::string label)
		{
			return std::make_shared<layr::Dense>(weight, bias, label);
		},
		py::arg("weight"),
		py::arg("bias") = nullptr,
		py::arg("label"));
	dense
		.def(py::init<teq::DimT,teq::DimT,
			layr::InitF<PybindT>,
			layr::InitF<PybindT>,
			const std::string&>(),
			py::arg("nunits"),
			py::arg("indim"),
			py::arg("weight_init") = layr::unif_xavier_init<PybindT>(1),
			py::arg("bias_init") = layr::zero_init<PybindT>(),
			py::arg("label"))
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
		py::arg("label"));
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
			py::arg("label"))
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
		py::arg("label"));
	conv
		.def(py::init<std::pair<teq::DimT,teq::DimT>,
			teq::DimT,teq::DimT,const std::string&,
			std::pair<teq::DimT,teq::DimT>>(),
			py::arg("filter_hw"),
			py::arg("in_ncol"),
			py::arg("out_ncol"),
			py::arg("label"),
			py::arg("zero_padding") = std::pair<teq::DimT,teq::DimT>{0, 0})
		.def("clone", &layr::Conv::clone, py::arg("prefix") = "")
		.def("paddings", &layr::Conv::get_padding);

	// seqmodel
	seqmodel
		.def(py::init<const std::string&>(),
			py::arg("label"))
		.def("clone", &layr::SequentialModel::clone, py::arg("prefix") = "")
		.def("add", &layr::SequentialModel::push_back)
		.def("get_layers", &layr::SequentialModel::get_layers);

	// mlptrainer
	mlptrainer
		.def(py::init<layr::SequentialModel&,
			eteq::iSession&,layr::ApproxF,
			teq::Shape,teq::Shape,
			trainer::NodeUnarF,trainer::TrainingContext>(),
			py::arg("model"), py::arg("sess"), py::arg("update"),
			py::arg("inshape"), py::arg("outshape"),
			py::arg("gradprocess") = trainer::NodeUnarF(pyrocnnet::identity),
			py::arg("ctx") = trainer::TrainingContext())
		.def("train",
			[](py::object self, py::array train_in, py::array expected_out)
			{
				eteq::ShapedArr<PybindT> xarr;
				eteq::ShapedArr<PybindT> yarr;
				xarr.data_ = pyrocnnet::arr2vec(xarr.shape_, train_in);
				yarr.data_ = pyrocnnet::arr2vec(yarr.shape_, expected_out);
				self.cast<trainer::MLPTrainer*>()->train(xarr, yarr);
			},
			"train internal variables")
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
			size_t, teq::DimT, size_t>(),
			py::arg("train_interval") = 5,
			py::arg("rand_action_prob") = 0.05,
			py::arg("discount_rate") = 0.95,
			py::arg("target_update_rate") = 0.01,
			py::arg("exploration_period") = 1000,
			py::arg("store_interval") = 5,
			py::arg("mini_batch_size") = 32,
			py::arg("max_exp") = 30000);
	dqntrainer
		.def(py::init<layr::SequentialModel&,eteq::iSession&,
			layr::ApproxF,trainer::DQNInfo,
			trainer::NodeUnarF,trainer::DQNTrainingContext>(),
			py::arg("model"), py::arg("sess"),
			py::arg("update"), py::arg("param"),
			py::arg("gradprocess") = trainer::NodeUnarF(pyrocnnet::identity),
			py::arg("ctx") = trainer::DQNTrainingContext())
		.def("action",
			[](py::object self, py::array input)
			{
				eteq::ShapedArr<PybindT> arr;
				arr.data_ = pyrocnnet::arr2vec(arr.shape_, input);
				return self.cast<trainer::DQNTrainer*>()->action(arr);
			},
			"get next action")
		.def("store", &trainer::DQNTrainer::store, "save observation, action, and reward")
		.def("train", &trainer::DQNTrainer::train, "train qnets")
		.def("error", &trainer::DQNTrainer::get_error, "get prediction error")
		.def("ntrained", &trainer::DQNTrainer::get_numtrained, "get number of iterations trained")
		.def("train_out",
			[](py::object self)
			{
				return self.cast<trainer::DQNTrainer*>()->train_out_;
			}, "get training node");

	// brbmtrainer
	brbmtrainer
		.def(py::init<
			layr::RBM&,
			eteq::iSession&,
			teq::DimT,
			PybindT,
			PybindT,
			trainer::ErrorF>(),
			py::arg("model"),
			py::arg("sess"),
			py::arg("batch_size"),
			py::arg("learning_rate"),
			py::arg("discount_factor"),
			py::arg("err_func"))
		.def("train",
			[](py::object self, py::array data)
			{
				auto trainer = self.cast<trainer::BernoulliRBMTrainer*>();
				teq::Shape shape;
				std::vector<PybindT> vec = pyrocnnet::arr2vec(shape, data);
				return trainer->train(vec);
			}, "train internal variables");

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
			[](py::object self, py::array x, size_t nepochs,
				std::function<void(size_t,size_t)> logger)
			{
				auto trainer = self.cast<trainer::DBNTrainer*>();
				teq::Shape shape;
				std::vector<PybindT> vec = pyrocnnet::arr2vec(shape, x);
				return trainer->pretrain(vec, nepochs, logger);
			},
			py::arg("x"),
			py::arg("nepochs") = 100,
			py::arg("logger") = std::function<void(size_t,size_t)>(),
			"pretrain internal rbms")
		.def("finetune",
			[](py::object self, py::array x, py::array y, size_t nepochs,
				std::function<void(size_t)> logger)
			{
				auto trainer = self.cast<trainer::DBNTrainer*>();
				teq::Shape shape;
				std::vector<PybindT> xvec = pyrocnnet::arr2vec(shape, x);
				std::vector<PybindT> yvec = pyrocnnet::arr2vec(shape, y);
				return trainer->finetune(xvec, yvec, nepochs, logger);
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
		.def("get_rms_momentum", &pyrocnnet::get_rms_momentum,
			py::arg("learning_rate") = 0.5,
			py::arg("discount_factor") = 0.99,
			py::arg("epsilon") = std::numeric_limits<PybindT>::epsilon())

		// inits
		.def("variable_from_init",
			[](layr::InitF<PybindT> init, std::vector<py::ssize_t> slist, std::string label)
			{
				return init(pyrocnnet::p2cshape(slist), label);
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
		.def("sigmoid", layr::sigmoid)
		.def("tanh", layr::tanh)
		.def("relu", layr::relu)
		.def("softmax", layr::softmax)
		.def("maxpool2d", layr::maxpool2d,
			py::arg("dims") = std::pair<teq::DimT,teq::DimT>{0, 1})
		.def("meanpool2d", layr::meanpool2d,
			py::arg("dims") = std::pair<teq::DimT,teq::DimT>{0, 1})
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
			});
};
