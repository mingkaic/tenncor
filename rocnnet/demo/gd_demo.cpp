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

#include "modl/model.hpp"
#include "modl/activations.hpp"

#include "rocnnet/trainer/mlp_trainer.hpp"

static std::vector<float> batch_generate (size_t n, size_t batchsize)
{
	size_t total = n * batchsize;

	// Specify the engine and distribution.
	std::mt19937 mersenne_engine(eteq::get_engine()());
	std::uniform_real_distribution<float> dist(0, 1);

	auto gen = std::bind(dist, mersenne_engine);
	std::vector<float> vec(total);
	std::generate(std::begin(vec), std::end(vec), gen);
	return vec;
}

static std::vector<float> avgevry2 (std::vector<float>& in)
{
	std::vector<float> out;
	for (size_t i = 0, n = in.size()/2; i < n; i++)
	{
		float val = (in.at(2*i) + in.at(2*i+1)) / 2;
		out.push_back(val);
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
		eteq::get_engine().seed(seedval);
	}

	uint8_t n_in = 10;
	uint8_t n_out = n_in / 2;
	std::vector<teq::DimT> n_outs = {9, n_out};

	modl::SequentialModel model("demo");
	model.push_back(std::make_shared<modl::Dense>(9, n_in,
		eqns::unif_xavier_init<PybindT>(1), eqns::zero_init<PybindT>(), "0"));
	model.push_back(modl::sigmoid());
	model.push_back(std::make_shared<modl::Dense>(n_out, 9,
		eqns::unif_xavier_init<PybindT>(1), eqns::zero_init<PybindT>(), "1"));
	model.push_back(modl::sigmoid());

	modl::SequentialModel untrained_model(model);
	modl::SeqModelptrT trained_model = nullptr;

	std::ifstream loadstr(loadpath);
	if (loadstr.is_open())
	{
		teq::TensT trained_roots;
		trained_model = std::static_pointer_cast<modl::SequentialModel>(
			modl::load_layer(loadstr, trained_roots, modl::seq_model_key, "demo"));
		logs::infof("model successfully loaded from file `%s`", loadpath.c_str());
		loadstr.close();
	}
	else
	{
		logs::warnf("model failed to loaded from file `%s`", loadpath.c_str());
		trained_model = std::make_shared<modl::SequentialModel>(model);
	}

	uint8_t n_batch = 3;
	size_t show_every_n = 500;
	eqns::ApproxF approx = [](const eqns::VarErrsT& leaves)
	{
		return eqns::sgd(leaves, 0.9); // learning rate = 0.9
	};
	dbg::InteractiveSession sess("localhost:50051");
	trainer::MLPTrainer trainer(model, sess, approx, n_batch);

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
		if (i % show_every_n == show_every_n - 1)
		{
			float* trained_err_res = trainer.error_->data();
			std::cout << "training " << i + 1 << '\n';
			std::cout << "trained error: " <<
				fmts::to_string(trained_err_res, trained_err_res + n_out) << '\n';
		}
		std::vector<float> batch = batch_generate(n_in, n_batch);
		std::vector<float> batch_out = avgevry2(batch);
		trainer.train(batch, batch_out);
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
		std::vector<float> batch = batch_generate(n_in, 1);
		std::vector<float> batch_out = avgevry2(batch);
		testin->assign(batch.data(), testin->shape());
		sess.update({testin->get_tensor().get()});

		float untrained_avgerr = 0;
		float trained_avgerr = 0;
		float pretrained_avgerr = 0;
		for (size_t i = 0; i < n_out; i++)
		{
			untrained_avgerr += std::abs(untrained_res[i] - batch_out[i]);
			trained_avgerr += std::abs(trained_res[i] - batch_out[i]);
			pretrained_avgerr += std::abs(pretrained_res[i] - batch_out[i]);
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
			if (modl::save_layer(savestr, model, {}))
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
