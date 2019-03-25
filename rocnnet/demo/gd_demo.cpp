#include <random>
#include <algorithm>
#include <iterator>
#include <ctime>
#include <iostream>
#include <fstream>

#include "pbm/save.hpp"

#include "flag/flag.hpp"

#include "ead/ead.hpp"

#include "rocnnet/eqns/activations.hpp"

#include "rocnnet/trainer/mlp_trainer.hpp"

static std::vector<float> batch_generate (size_t n, size_t batchsize)
{
	size_t total = n * batchsize;

	// Specify the engine and distribution.
	std::mt19937 mersenne_engine(ead::get_engine()());
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

int main (int argc, char** argv)
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
			"rocnnet/pretrained/gdmodel.pbx"), "filename to load pretrained model");

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
		ead::get_engine().seed(seedval);
	}

	uint8_t n_in = 10;
	uint8_t n_out = n_in / 2;

	std::vector<modl::LayerInfo> hiddens = {
		// use same sigmoid in static memory once models copy is established
		modl::LayerInfo{9, age::sigmoid<float>},
		modl::LayerInfo{n_out, age::sigmoid<float>},
	};
	auto brain = std::make_shared<modl::MLP>(n_in, hiddens, "brain");
	auto untrained_brain = std::make_shared<modl::MLP>(*brain);
	auto pretrained_brain = std::make_shared<modl::MLP>(*brain);
	std::ifstream loadstr(loadpath);
	if (loadstr.is_open())
	{
		modl::load(loadstr, pretrained_brain.get());
		loadstr.close();
	}

	uint8_t n_batch = 3;
	size_t show_every_n = 500;
	eqns::ApproxFuncT approx = [](ead::NodeptrT<float>& root, eqns::VariablesT leaves)
	{
		return eqns::sgd(root, leaves, 0.9); // learning rate = 0.9
	};
	ead::Session<float> sess;
	MLPTrainer trainer(brain, sess, approx, n_batch);

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

	ead::VarptrT<float> testin = ead::make_variable_scalar<float>(
		0, ade::Shape({n_in}), "testin");
	auto untrained_out = (*untrained_brain)(testin);
	auto trained_out = (*brain)(testin);
	auto pretrained_out = (*pretrained_brain)(testin);
	sess.track(untrained_out);
	sess.track(trained_out);
	sess.track(pretrained_out);

	float* untrained_res = untrained_out->data();
	float* trained_res = trained_out->data();
	float* pretrained_res = pretrained_out->data();
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
			modl::save(savestr, trained_out->get_tensor(), brain.get());
			savestr.close();
		}
	}

	google::protobuf::ShutdownProtobufLibrary();

	return exit_status;
}
