#include <random>
#include <algorithm>
#include <iterator>
#include <ctime>
#include <iostream>
#include <fstream>

#include "flag/flag.hpp"

#include "tenncor/eteq/eteq.hpp"

#include "tenncor/tenncor.hpp"

// #include "dbg/peval/plugin_eval.hpp"
// #include "dbg/peval/emit/emitter.hpp"
#include "dbg/compare/equal.hpp"

#include "tenncor/layr/layer.hpp"

#include "tenncor/trainer/trainer.hpp"

static trainer::ShapedArr<float> batch_generate (teq::DimT n, teq::DimT batchsize)
{
	// Specify the engine and distribution.
	std::mt19937 mersenne_engine(global::get_generator()->unif_int(
		std::numeric_limits<int64_t>::min(),
		std::numeric_limits<int64_t>::max()));
	std::uniform_real_distribution<float> dist(0, 1);

	auto gen = std::bind(dist, mersenne_engine);
	trainer::ShapedArr<float> out(teq::Shape({n, batchsize}));
	std::generate(std::begin(out.data_), std::end(out.data_), gen);
	return out;
}

static trainer::ShapedArr<float> avgevry2 (trainer::ShapedArr<float>& in)
{
	trainer::ShapedArr<float> out(teq::Shape({
		static_cast<teq::DimT>(in.shape_.at(0) / 2),
		in.shape_.at(1)}));
	for (size_t i = 0, n = in.data_.size() / 2; i < n; i++)
	{
		out.data_[i] = (in.data_.at(2*i) + in.data_.at(2*i+1)) / 2;
	}
	return out;
}

static inline eteq::ETensor sigmoid (const eteq::ETensor& x)
{
	return tenncor().sigmoid(x);
}

int main (int argc, const char** argv)
{
	GOOGLE_PROTOBUF_VERIFY_VERSION;

	auto& tc = tenncor();

	bool seed;
	size_t seedval;
	size_t n_train;
	size_t n_test;
	std::string savepath;
	std::string loadpath;

	// size_t default_seed = static_cast<size_t>(
	// 	std::chrono::high_resolution_clock::now().
	// 		time_since_epoch().count());
	size_t default_seed = 0;

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
		global::seed(seedval);
	}

	uint8_t n_in = 10;
	uint8_t n_hid = 9;
	uint8_t n_out = n_in / 2;

	eteq::ETensor trained_model = tc.layer.link({
		tc.layer.dense<float>(teq::Shape({n_in}), {n_hid},
			tc.layer.unif_xavier_init<float>(1), tc.layer.zero_init<float>()),
		tc.layer.bind<float>(sigmoid),
		tc.layer.dense<float>(teq::Shape({n_hid}), {n_out},
			tc.layer.unif_xavier_init<float>(1), tc.layer.zero_init<float>()),
		tc.layer.bind<float>(sigmoid),
	});
	eteq::ETensor untrained_model = layr::deep_clone(trained_model);
	eteq::ETensor pretrained_model = layr::deep_clone(trained_model);

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
		onnx::TensptrIdT ids;
		pretrained_model = tcr::load_model(ids, pb_model)[0];
		global::infof("model successfully loaded from file `%s`", loadpath.c_str());
		loadstr.close();
	}
	catch (...)
	{
		global::warnf("model failed to loaded from file `%s`", loadpath.c_str());
	}

	uint8_t n_batch = 3;
	size_t show_every_n = 500;
	layr::ApproxF<float> approx =
		[&](const eteq::ETensor& error,
			const eteq::EVariablesT<float>& leaves)
		{
			return tc.approx.sgd<float>(error, leaves, 0.9); // learning rate = 0.9
		};
	// emit::Emitter emitter("localhost:50051");
	// auto eval = new dbg::PlugableEvaluator();
	// teq::set_eval(eval, global::context());
	// eval->add_plugin(emitter);
	{

	// jobs::ScopeGuard defer(
	// 	[&emitter]
	// 	{
	// 		// 10 seconds
	// 		std::chrono::time_point<std::chrono::system_clock> deadline =
	// 			std::chrono::system_clock::now() +
	// 			std::chrono::seconds(10);
	// 		emitter.join_then_stop(deadline);
	// 	});

	auto train_input = eteq::make_variable_scalar<float>(0, teq::Shape({n_in, n_batch}));
	auto train_exout = eteq::make_variable_scalar<float>(0, teq::Shape({n_out, n_batch}));
	auto train_err = trainer::apply_update<float>({trained_model}, approx,
		[&](const eteq::ETensorsT& models)
		{
			return tc.error.sqr_diff(train_exout,
				layr::connect(models.front(), train_input));
		});

	eteq::EVariable<float> testin = eteq::make_variable_scalar<float>(
		0, teq::Shape({n_in}), "testin");
	auto untrained_out = layr::connect(untrained_model, testin);
	auto trained_out = layr::connect(trained_model, testin);
	auto pretrained_out = layr::connect(pretrained_model, testin);

	hone::optimize("cfg/optimizations.json");

	// train mlp to output input
	start = std::clock();
	for (size_t i = 0; i < n_train; i++)
	{
		trainer::ShapedArr<float> batch = batch_generate(n_in, n_batch);
		trainer::ShapedArr<float> batch_out = avgevry2(batch);
		train_input->assign(batch.data_.data(), batch.shape_);
		train_exout->assign(batch_out.data_.data(), batch_out.shape_);
		float* data = train_err.calc<float>();
		if (i % show_every_n == show_every_n - 1)
		{
			float* data_end = data + train_err->shape().n_elems();
			float ferr = std::accumulate(data, data_end, 0.f);
			std::cout << "training " << i + 1 << '\n';
			std::cout << "trained error: "
				<< fmts::to_string(data, data_end) << "~" << ferr << '\n';
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

	for (size_t i = 0; i < n_test; i++)
	{
		if (i % show_every_n == show_every_n - 1)
		{
			std::cout << "testing " << i + 1 << '\n';
		}
		trainer::ShapedArr<float> batch = batch_generate(n_in, 1);
		trainer::ShapedArr<float> batch_out = avgevry2(batch);
		testin->assign(batch.data_.data(), batch.shape_);
		float* untrained_res = untrained_out.calc<float>();
		float* trained_res = trained_out.calc<float>();
		float* pretrained_res = pretrained_out.calc<float>();

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
	std::cout << "is structurally equal?: " << is_equal(trained_out, pretrained_out) << "\n";
	std::cout << "% data equal: " << percent_dataeq(trained_out, pretrained_out) << std::endl;

	// try to save
	if (exit_status == 0)
	{
		std::ofstream savestr(savepath);
		if (savestr.is_open())
		{
			onnx::ModelProto pb_model;
			tcr::save_model(pb_model, {trained_model});
			if (pb_model.SerializeToOstream(&savestr))
			{
				global::infof("successfully saved model to `%s`", savepath.c_str());
			}
			savestr.close();
		}
		else
		{
			global::warnf("failed to save model to `%s`", savepath.c_str());
		}
	}

	}

	google::protobuf::ShutdownProtobufLibrary();
	return exit_status;
}
