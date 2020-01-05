#include <random>
#include <algorithm>
#include <iterator>
#include <ctime>
#include <iostream>
#include <fstream>

#include "flag/flag.hpp"

#include "eteq/eteq.hpp"
#include "eteq/optimize.hpp"

#include "dbg/psess/plugin_sess.hpp"
#include "dbg/psess/emit/emitter.hpp"

#include "layr/api.hpp"
#include "layr/trainer/sgd.hpp"

static teq::ShapedArr<PybindT> batch_generate (teq::DimT n, teq::DimT batchsize)
{
	// Specify the engine and distribution.
	std::mt19937 mersenne_engine(eigen::get_engine()());
	std::uniform_real_distribution<float> dist(0, 1);

	auto gen = std::bind(dist, mersenne_engine);
	teq::ShapedArr<PybindT> out(teq::Shape({n, batchsize}));
	std::generate(std::begin(out.data_), std::end(out.data_), gen);
	return out;
}

static teq::ShapedArr<PybindT> avgevry2 (teq::ShapedArr<PybindT>& in)
{
	teq::ShapedArr<PybindT> out(teq::Shape({
		static_cast<teq::DimT>(in.shape_.at(0) / 2),
		in.shape_.at(1)}));
	for (size_t i = 0, n = in.data_.size() / 2; i < n; i++)
	{
		out.data_[i] = (in.data_.at(2*i) + in.data_.at(2*i+1)) / 2;
	}
	return out;
}

int main (int argc, const char** argv)
{
	GOOGLE_PROTOBUF_VERIFY_VERSION;

	bool seed;
	size_t seedval;
	size_t n_train;
	size_t n_test;
	std::string savepath;
	std::string loadpath;

	size_t default_seed = static_cast<size_t>(
		std::chrono::high_resolution_clock::now().
			time_since_epoch().count());

	flag::FlagSet flags("gd_demo");
	flags.add_flags()
		("seed", flag::opt::bool_switch(&seed)->default_value(true), "whether to seed or not")
		("seedval", flag::opt::value<size_t>(&seedval)->default_value(default_seed),
			"random seed value")
		("n_train", flag::opt::value<size_t>(&n_train)->default_value(3000),
			"number of times to train")
		("n_test", flag::opt::value<size_t>(&n_test)->default_value(500),
			"number of times to test")
		("save", flag::opt::value<std::string>(&savepath)->default_value(""),
			"filename to save model")
		("load", flag::opt::value<std::string>(&loadpath)->default_value(
			"models/gd.onnx"), "filename to load pretrained model");

	int exit_status = 0;
	std::clock_t start;
	float duration;

	if (false == flags.parse(argc, argv))
	{
		return 1;
	}

	if (seed)
	{
		std::cout << "seeding " << seedval << '\n';
		eigen::get_engine().seed(seedval);
	}

	uint8_t n_in = 10;
	uint8_t n_hid = 9;
	uint8_t n_out = n_in / 2;
	std::vector<teq::DimT> n_outs = {9, n_out};

	eteq::ELayer<PybindT> model = layr::link<PybindT>({
		layr::dense<PybindT>(teq::Shape({n_in}), {n_hid},
			layr::unif_xavier_init<PybindT>(1), layr::zero_init<PybindT>()),
		layr::bind(layr::UnaryF<PybindT>(tenncor::sigmoid<PybindT>)),
		layr::dense<PybindT>(teq::Shape({n_hid}), {n_out},
			layr::unif_xavier_init<PybindT>(1), layr::zero_init<PybindT>()),
		layr::bind(layr::UnaryF<PybindT>(tenncor::sigmoid<PybindT>)),
	});
	eteq::ELayer<PybindT> untrained_model = model.deep_clone();
	eteq::ELayer<PybindT> trained_model = model.deep_clone();

	std::ifstream loadstr(loadpath);
	try
	{
		if (false == loadstr.is_open())
		{
			throw std::exception();
		}
		onnx::ModelProto pb_model;
		if (false == pb_model.ParseFromIstream(&loadstr))
		{
			throw std::exception();
		}
		trained_model = eteq::load_layers<PybindT>(pb_model)[0];
		logs::infof("model successfully loaded from file `%s`", loadpath.c_str());
		loadstr.close();
	}
	catch (...)
	{
		logs::warnf("model failed to loaded from file `%s`", loadpath.c_str());
	}

	uint8_t n_batch = 3;
	size_t show_every_n = 500;
	layr::ApproxF<PybindT> approx =
		[](const layr::VarErrsT<PybindT>& leaves)
		{
			return layr::sgd<PybindT>(leaves, 0.9); // learning rate = 0.9
		};
	emit::Emitter emitter("localhost:50051");
	dbg::PluginSession sess;
	sess.plugins_.push_back(emitter);
	{

	jobs::ScopeGuard defer(
		[&emitter]
		{
			// 10 seconds
			std::chrono::time_point<std::chrono::system_clock> deadline =
				std::chrono::system_clock::now() +
				std::chrono::seconds(10);
			emitter.join_then_stop(deadline);
		});

	auto train_input = eteq::make_variable_scalar<PybindT>(0, teq::Shape({n_in, n_batch}));
	auto train_output = eteq::make_variable_scalar<PybindT>(0, teq::Shape({n_out, n_batch}));
	auto train = trainer::sgd(model, sess,
		eteq::ETensor<PybindT>(train_input),
		eteq::ETensor<PybindT>(train_output), approx);

	eteq::VarptrT<float> testin = eteq::make_variable_scalar<float>(
		0, teq::Shape({n_in}), "testin");
	auto untrained_out = untrained_model.connect(eteq::ETensor<PybindT>(testin));
	auto out = model.connect(eteq::ETensor<PybindT>(testin));
	auto trained_out = trained_model.connect(eteq::ETensor<PybindT>(testin));
	sess.track({untrained_out, out, trained_out});

	auto rules = eteq::parse_file<PybindT>("cfg/optimizations.rules");
	eteq::optimize<PybindT>(sess, rules);

	// train mlp to output input
	start = std::clock();
	for (size_t i = 0; i < n_train; i++)
	{
		teq::ShapedArr<PybindT> batch = batch_generate(n_in, n_batch);
		teq::ShapedArr<PybindT> batch_out = avgevry2(batch);
		train_input->assign(batch);
		train_output->assign(batch_out);
		auto err = train();
		if (i % show_every_n == show_every_n - 1)
		{
			float ferr = std::accumulate(err.data_.begin(), err.data_.end(), 0.f);
			std::cout << "training " << i + 1 << '\n';
			std::cout << "trained error: " << fmts::to_string(
				err.data_.begin(), err.data_.end()) << "~" << ferr << '\n';
		}
	}
	duration = (std::clock() - start) / (float) CLOCKS_PER_SEC;
	std::cout << "training time: " << duration << " seconds" << '\n';

	// exit code:
	//	0 = fine
	//	1 = training error rate is wrong
	float untrained_err = 0;
	float trained_err = 0;
	float pretrained_err = 0;

	float* untrained_res = (PybindT*) untrained_out->data();
	float* trained_res = (PybindT*) out->data();
	float* pretrained_res = (PybindT*) trained_out->data();
	for (size_t i = 0; i < n_test; i++)
	{
		if (i % show_every_n == show_every_n - 1)
		{
			std::cout << "testing " << i + 1 << '\n';
		}
		teq::ShapedArr<PybindT> batch = batch_generate(n_in, 1);
		teq::ShapedArr<PybindT> batch_out = avgevry2(batch);
		testin->assign(batch);
		sess.update();

		float untrained_avgerr = 0;
		float trained_avgerr = 0;
		float pretrained_avgerr = 0;
		for (size_t i = 0; i < n_out; i++)
		{
			untrained_avgerr += std::abs(untrained_res[i] - batch_out.data_[i]);
			trained_avgerr += std::abs(trained_res[i] - batch_out.data_[i]);
			pretrained_avgerr += std::abs(pretrained_res[i] - batch_out.data_[i]);
		}
		untrained_err += untrained_avgerr / n_out;
		trained_err += trained_avgerr / n_out;
		pretrained_err += pretrained_avgerr / n_out;
	}
	untrained_err /= (float) n_test;
	trained_err /= (float) n_test;
	pretrained_err /= (float) n_test;
	std::cout << "untrained mlp error rate: " << untrained_err * 100 << "%\n";
	std::cout << "trained mlp error rate: " << trained_err * 100 << "%\n";
	std::cout << "pretrained mlp error rate: " << pretrained_err * 100 << "%\n";

	// try to save
	if (exit_status == 0)
	{
		std::ofstream savestr(savepath);
		if (savestr.is_open())
		{
			onnx::ModelProto pb_model;
			eteq::save_layers<PybindT>(pb_model, {model});
			if (pb_model.SerializeToOstream(&savestr))
			{
				logs::infof("successfully saved model to `%s`", savepath.c_str());
			}
			savestr.close();
		}
		else
		{
			logs::warnf("failed to save model to `%s`", savepath.c_str());
		}
	}

	}

	google::protobuf::ShutdownProtobufLibrary();
	return exit_status;
}
