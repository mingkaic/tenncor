#include <random>
#include <algorithm>
#include <iterator>
#include <ctime>
#include <iostream>
#include <fstream>

#include "flag/flag.hpp"

#include "eteq/eteq.hpp"
#include "eteq/parse.hpp"

#include "dbg/grpc/session.hpp"

#include "layr/seqmodel.hpp"
#include "layr/ulayer.hpp"

#include "rocnnet/trainer/sgd_trainer.hpp"

static eteq::ShapedArr<PybindT> batch_generate (teq::DimT n, teq::DimT batchsize)
{
	// Specify the engine and distribution.
	std::mt19937 mersenne_engine(eigen::get_engine()());
	std::uniform_real_distribution<float> dist(0, 1);

	auto gen = std::bind(dist, mersenne_engine);
	eteq::ShapedArr<PybindT> out(teq::Shape({n, batchsize}));
	std::generate(std::begin(out.data_), std::end(out.data_), gen);
	return out;
}

static eteq::ShapedArr<PybindT> avgevry2 (eteq::ShapedArr<PybindT>& in)
{
	eteq::ShapedArr<PybindT> out(teq::Shape({
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
			"models/gdmodel.pbx"), "filename to load pretrained model");

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
	uint8_t n_out = n_in / 2;
	std::vector<teq::DimT> n_outs = {9, n_out};

	layr::SequentialModel model("demo");
	model.push_back(std::make_shared<layr::Dense>(9, teq::Shape({n_in}),
		layr::unif_xavier_init<PybindT>(1),
		layr::zero_init<PybindT>(), nullptr, "0"));
	model.push_back(layr::sigmoid());
	model.push_back(std::make_shared<layr::Dense>(n_out, teq::Shape({9}),
		layr::unif_xavier_init<PybindT>(1),
		layr::zero_init<PybindT>(), nullptr, "1"));
	model.push_back(layr::sigmoid());

	layr::SequentialModel untrained_model(model);
	layr::SeqModelptrT trained_model = nullptr;

	std::ifstream loadstr(loadpath);
	if (loadstr.is_open())
	{
		teq::TensptrsT trained_roots;
		trained_model = std::static_pointer_cast<layr::SequentialModel>(
			layr::load_layer(loadstr, trained_roots, layr::seq_model_key, "demo"));
		logs::infof("model successfully loaded from file `%s`", loadpath.c_str());
		loadstr.close();
	}
	else
	{
		logs::warnf("model failed to loaded from file `%s`", loadpath.c_str());
		trained_model = std::make_shared<layr::SequentialModel>(model);
	}

	uint8_t n_batch = 3;
	size_t show_every_n = 500;
	layr::ApproxF approx = [](const layr::VarErrsT& leaves)
	{
		return layr::sgd(leaves, 0.9); // learning rate = 0.9
	};
	dbg::InteractiveSession sess("localhost:50051");

	auto train_input = eteq::make_variable_scalar<PybindT>(0, teq::Shape({n_in, n_batch}));
	auto train_output = eteq::make_variable_scalar<PybindT>(0, teq::Shape({n_out, n_batch}));
	auto train = trainer::sgd_train(model, sess,
		eteq::convert_to_node(train_input), eteq::convert_to_node(train_output), approx);

	eteq::VarptrT<float> testin = eteq::make_variable_scalar<float>(
		0, teq::Shape({n_in}), "testin");
	auto untrained_out = untrained_model.connect(testin);
	auto out = model.connect(testin);
	auto trained_out = trained_model->connect(testin);
	sess.track({
		untrained_out->get_tensor(),
		out->get_tensor(),
		trained_out->get_tensor(),
	});

	opt::OptCtx rules = eteq::parse_file<PybindT>("cfg/optimizations.rules");
	sess.optimize(rules);

	// train mlp to output input
	start = std::clock();
	for (size_t i = 0; i < n_train; i++)
	{
		eteq::ShapedArr<PybindT> batch = batch_generate(n_in, n_batch);
		eteq::ShapedArr<PybindT> batch_out = avgevry2(batch);
		train_input->assign(batch);
		train_output->assign(batch_out);
		auto err = train();
		if (i % show_every_n == show_every_n - 1)
		{
			std::cout << "training " << i + 1 << '\n';
			std::cout << "trained error: " <<
				fmts::to_string(err.data_.begin(), err.data_.end()) << '\n';
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

	float* untrained_res = untrained_out->data();
	float* trained_res = out->data();
	float* pretrained_res = trained_out->data();
	for (size_t i = 0; i < n_test; i++)
	{
		if (i % show_every_n == show_every_n - 1)
		{
			std::cout << "testing " << i + 1 << '\n';
		}
		eteq::ShapedArr<PybindT> batch = batch_generate(n_in, 1);
		eteq::ShapedArr<PybindT> batch_out = avgevry2(batch);
		testin->assign(batch.data_.data(), batch.shape_);
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
			if (layr::save_layer(savestr, model, {}))
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

	// 10 seconds
	std::chrono::time_point<std::chrono::system_clock> deadline =
		std::chrono::system_clock::now() +
		std::chrono::seconds(30);
	sess.join_then_stop(deadline);

	google::protobuf::ShutdownProtobufLibrary();

	return exit_status;
}
