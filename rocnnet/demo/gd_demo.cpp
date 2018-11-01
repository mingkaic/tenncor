#include <random>
#include <algorithm>
#include <iterator>
#include <ctime>
#include <iostream>
#include <fstream>

#include "dbg/ade.hpp"

#include "pbm/graph.hpp"

#include "rocnnet/eqns/activations.hpp"

#include "rocnnet/modl/gd_trainer.hpp"

// user-side generator
static std::default_random_engine rnd_device(std::time(NULL));

static std::vector<double> batch_generate (size_t n, size_t batchsize)
{
	size_t total = n * batchsize;

	// Specify the engine and distribution.
	std::mt19937 mersenne_engine(rnd_device());
	std::uniform_real_distribution<double> dist(0, 1);

	auto gen = std::bind(dist, mersenne_engine);
	std::vector<double> vec(total);
	std::generate(std::begin(vec), std::end(vec), gen);
	return vec;
}

static std::vector<double> avgevry2 (std::vector<double>& in)
{
	std::vector<double> out;
	for (size_t i = 0, n = in.size()/2; i < n; i++)
	{
		double val = (in.at(2*i) + in.at(2*i+1)) / 2;
		out.push_back(val);
	}
	return out;
}

int main (int argc, char** argv)
{
	GOOGLE_PROTOBUF_VERIFY_VERSION;
	int exit_status = 0;
	std::clock_t start;
	double duration;
	std::string outdir = "data";
	size_t n_train = 3000;
	size_t n_test = 500;
	// size_t seed_val;
	bool seed = false;
	bool save = false;
	std::string serialname = "gd_test.pbx";
	std::string serialpath = outdir + "/" + serialname;

	// todo: hookup option parser

	if (seed)
	{
		// rnd_device.seed(seed_val);
		// seed_generator(seed_val);
	}
	else
	{
		// seed by time
	}

	uint8_t n_in = 10;
	uint8_t n_out = n_in / 2;
	std::vector<LayerInfo> hiddens = {
		// use same sigmoid in static memory once models copy is established
		LayerInfo{9, sigmoid},
		LayerInfo{n_out, sigmoid}
	};
	MLP brain(n_in, hiddens, "brain");
	MLP untrained_brain(brain, "untrained_");
	// todo: implement
	// MLP pretrained_brain(serialpath);

	uint8_t n_batch = 3;
	size_t show_every_n = 500;
	ApproxFuncT approx = [](llo::DataNode& root, std::vector<llo::DataNode> leaves)
	{
		return sgd(root, leaves, 0.9); // learning rate = 0.9
	};
	GDTrainer trainer(brain, approx, n_batch, "gdn");

	PrettyEquation peq;
	size_t i = 0;
	for (auto deltas : trainer.updates_)
	{
		peq.labels_[deltas.first->inner().get()] = ade::sprintf("var%d", i);
		++i;
	}
	peq.labels_[trainer.expected_out_.tensor_.get()] = "expected_out";
	peq.labels_[trainer.error_.tensor_.get()] = "error";
	peq.labels_[trainer.train_in_.tensor_.get()] = "train_in";
	peq.print(std::cout, trainer.error_.tensor_);
	std::cout << "\n";
#if 0
	for (auto deltas : trainer.updates_)
	{
		peq.print(std::cout, deltas.second.tensor_);
		std::cout << "\n";
	}
#endif

	std::vector<double> batch = batch_generate(n_in, n_batch);
	std::vector<double> batch_out = avgevry2(batch);
	trainer.train_in_ = batch;
	trainer.expected_out_ = batch_out;

	// train mlp to output input
	start = std::clock();
	for (size_t i = 0; i < n_train; i++)
	{
		if (i % show_every_n == show_every_n-1)
		{
			llo::GenericData trained_derr = trainer.error_.data(llo::DOUBLE);
			double* trained_err_res = (double*) trained_derr.data_.get();
			std::cout << "training " << i+1 << std::endl;
			std::cout << "trained error: " << ade::to_string(trained_err_res, trained_err_res + n_out) << std::endl;
		}
		std::vector<double> batch = batch_generate(n_in, n_batch);
		std::vector<double> batch_out = avgevry2(batch);
		trainer.train(batch, batch_out);
	}
	duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
	std::cout << "training time: " << duration << " seconds" << std::endl;

	// exit code:
	//	0 = fine
	//	1 = training error rate is wrong
	double untrained_err = 0;
	double trained_err = 0;
	double pretrained_err = 0;

	llo::PlaceHolder<double> testin(ade::Shape({n_in}));
	auto untrained_out = untrained_brain(testin);
	auto trained_out = brain(testin);
	// auto pretrained_out = pretrained_brain(testin);
	for (size_t i = 0; i < n_test; i++)
	{
		if (i % show_every_n == show_every_n-1)
		{
			std::cout << "testing " << i+1 << "\n";
		}
		std::vector<double> batch = batch_generate(n_in, 1);
		std::vector<double> batch_out = avgevry2(batch);
		testin = batch;

		llo::GenericData untrained_data = untrained_out.data(llo::DOUBLE);
		llo::GenericData trained_data = trained_out.data(llo::DOUBLE);
		// llo::GenericData pretrained_data = pretrained_out.data(llo::DOUBLE);

		double* untrained_res = (double*) untrained_data.data_.get();
		double* trained_res = (double*) trained_data.data_.get();
		// double* pretrained_res = (double*) pretrained_data.data_.get();

		double untrained_avgerr = 0;
		double trained_avgerr = 0;
		// double pretrained_avgerr = 0;
		for (size_t i = 0; i < n_out; i++)
		{
			untrained_avgerr += std::abs(untrained_res[i] - batch_out[i]);
			trained_avgerr += std::abs(trained_res[i] - batch_out[i]);
			// pretrained_avgerr += std::abs(pretrained_res[i] - batch_out[i]);
		}
		untrained_err += untrained_avgerr / n_out;
		trained_err += trained_avgerr / n_out;
		// pretrained_err += pretrained_avgerr / n_out;
	}
	untrained_err /= (double) n_test;
	trained_err /= (double) n_test;
	// pretrained_err /= (double) n_test;
	std::cout << "untrained mlp error rate: " << untrained_err * 100 << "%\n";
	std::cout << "trained mlp error rate: " << trained_err * 100 << "%\n";
	// std::cout << "pretrained mlp error rate: " << pretrained_err * 100 << "%\n";

	if (exit_status == 0 && save)
	{
		// trained_gdn.save(serialpath, "gd_demo");
	}

// #ifdef CSV_RCD
// if (rocnnet_record::record_status::rec_good)
// {
// 	static_cast<rocnnet_record::csv_record*>(rocnnet_record::record_status::rec.get())->
// 		to_csv<double>(trained_gdn.get_error());
// }
// #endif /* CSV_RCD */

	google::protobuf::ShutdownProtobufLibrary();

	return exit_status;
}