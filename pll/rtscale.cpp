// Weigh the runtime of each opcode
#include <fstream>
#include <chrono>
#include <limits>

#include "flag/flag.hpp"

#include "ead/generated/api.hpp"
#include "ead/generated/opcode.hpp"
#include "ead/functor.hpp"

#include "pll/weights.pb.h"

#define TIME(action)\
std::chrono::high_resolution_clock::time_point start =\
	std::chrono::high_resolution_clock::now();\
action;\
stat = std::chrono::duration_cast<std::chrono::nanoseconds>(\
	std::chrono::high_resolution_clock::now() - start).count();

double softplus (double x)
{
	return std::log(1 + std::exp(x));
}

int main (int argc, const char** argv)
{
	std::string writepath;
	flag::FlagSet flags("rt_anubis");
	flags.add_flags()
		("target", flag::opt::value<std::string>(&writepath),
			"filename of json to write weights to");

	if (false == flags.parse(argc, argv))
	{
		return 1;
	}

	logs::get_logger().set_log_level(logs::INFO);

	std::unordered_map<std::string,size_t> stats;
	size_t mean_stat = 0;//,
		// max_stat = 0,
		// min_stat = std::numeric_limits<size_t>::max();
	for (size_t i = 0; i < age::_N_GENERATED_OPCODES; ++i)
	{
		size_t stat;
		auto opcode = (age::_GENERATED_OPCODE) i;
		ade::Opcode op{age::name_op(opcode), opcode};
		logs::infof("weighing operation %s", op.name_.c_str());
		switch (i)
		{
			// elementary unary
			case age::ABS:
			case age::NEG:
			case age::SIN:
			case age::COS:
			case age::TAN:
			case age::EXP:
			case age::LOG:
			case age::SQRT:
			case age::ROUND:
			case age::SIGMOID:
			case age::SIGMOID_GRAD:
			case age::TANH:
			case age::SQUARE:
			case age::CUBE:
			{
				auto var = ead::make_constant_scalar<float>(
					0.5, ade::Shape({56, 57, 58}));
				auto f = ead::make_functor<float>(op, {
					ead::identity_map(var)});
				TIME(f->update())
			}
				break;

			// elementary binary
			case age::POW:
			case age::ADD:
			case age::SUB:
			case age::MUL:
			case age::DIV:
			case age::MIN:
			case age::MAX:
			case age::EQ:
			case age::NEQ:
			case age::LT:
			case age::GT:
			case age::RAND_UNIF:
			{
				auto var = ead::make_constant_scalar<float>(
					0.5, ade::Shape({56, 57, 58}));
				auto f = ead::make_functor<float>(op, {
					ead::identity_map(var), ead::identity_map(var)});
				TIME(f->update())
			}
				break;

			// reductions
			case age::REDUCE_SUM:
			case age::REDUCE_PROD:
			case age::REDUCE_MIN:
			case age::REDUCE_MAX:
			{
				auto var = ead::make_constant_scalar<float>(
					0.5, ade::Shape({56, 57, 58}));
				auto f = ead::make_functor<float>(op, {
					ead::reduce_map(var, 1, 1)});
				TIME(f->update())
			}
				break;

			// other stuff
			case age::PERMUTE:
			{
				auto var = ead::make_constant_scalar<float>(
					0.5, ade::Shape({56, 57, 58}));
				auto f = ead::make_functor<float>(op, {
					ead::permute_map(var, {2, 0, 1})});
				TIME(f->update())
			}
				break;

			case age::EXTEND:
			{
				auto var = ead::make_constant_scalar<float>(
					0.5, ade::Shape({56, 58}));
				auto f = ead::make_functor<float>(op, {
					ead::extend_map(var, 2, {57})});
				TIME(f->update())
			}
				break;

			case age::SLICE:
			{
				auto var = ead::make_constant_scalar<float>(
					0.5, ade::Shape({56, 57, 58}));
				auto f = ead::make_functor<float>(op, {
					ead::slice_map(var, 2, 2, 2)});
				TIME(f->update())
			}
				break;

			case age::MATMUL:
			{
				auto a = ead::make_constant_scalar<float>(
					0.3, ade::Shape({253, 255}));
				auto b = ead::make_constant_scalar<float>(
					0.6, ade::Shape({254, 253}));
				auto f = tenncor::matmul(a, b);
				TIME(f->update())
			}
				break;

			case age::CONV:
			{
				auto img = ead::make_constant_scalar<float>(
					0.3, ade::Shape({254, 255}));
				auto kern = ead::make_constant_scalar<float>(
					0.6, ade::Shape({5, 7}));
				auto f = tenncor::convolution(img, kern, {0, 1});
				TIME(f->update())
			}
				break;

			case age::PAD:
			{
				auto var = ead::make_constant_scalar<float>(
					0.5, ade::Shape({56, 57, 58}));
				auto f = ead::make_functor<float>(op, {
					ead::pad_map(var, {3, 4}, 2)});
				TIME(f->update())
			}
				break;

			case age::SELECT:
			{
				ade::Shape shape({56, 57, 58});
				size_t n = shape.n_elems();
				std::vector<float> data;
				data.reserve(n);
				for (size_t i = 0; i < n; ++i)
				{
					data.push_back(i % 2);
				}
				auto cond = ead::make_constant<float>(data.data(), shape);
				auto a = ead::make_constant_scalar<float>(0.3, shape);
				auto b = ead::make_constant_scalar<float>(0.6, shape);
				auto f = tenncor::if_then_else(cond, a, b);
				TIME(f->update())
			}
				break;

			case age::CONV_IMG_GRAD:
			case age::CONV_KRN_GRAD:
			default:
				continue;
		}
		mean_stat += stat;
		// max_stat = std::max(max_stat, stat);
		// min_stat = std::min(min_stat, stat);
		stats.emplace(op.name_, stat);
	}

	double mean = (double) mean_stat / stats.size();

	// normalize stats by mean
	weights::OpWeights opweights;
	opweights.set_label("ead_weights");
	::google::protobuf::Map< ::std::string,double>* weights =
		opweights.mutable_weights();
	for (auto& op : stats)
	{
		double value = softplus((op.second - mean) / (
			mean + std::numeric_limits<double>::epsilon()));
		weights->insert({op.first, value});
	}

	logs::infof("writing to %s", writepath.c_str());
	std::fstream out(writepath,
		std::ios::out | std::ios::trunc | std::ios::binary);
	if (out.is_open())
	{
		logs::infof("opened %s", writepath.c_str());
		if (opweights.SerializeToOstream(&out))
		{
			logs::infof("done writing to %s", writepath.c_str());
		}
		out.close();
	}

	return 0;
}
