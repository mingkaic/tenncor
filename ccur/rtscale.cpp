// Weigh the runtime of each opcode
#include <fstream>
#include <chrono>
#include <limits>

#include "flag/flag.hpp"

#include "eigen/generated/opcode.hpp"

#include "eteq/generated/api.hpp"
#include "eteq/functor.hpp"

#include "ccur/weights.pb.h"

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
	for (size_t i = 0; i < egen::_N_GENERATED_OPCODES; ++i)
	{
		size_t stat;
		auto opcode = (egen::_GENERATED_OPCODE) i;
		teq::Opcode op{egen::name_op(opcode), opcode};
		logs::infof("weighing operation %s", op.name_.c_str());
		switch (i)
		{
			// elementary unary
			case egen::ABS:
			case egen::NEG:
			case egen::SIN:
			case egen::COS:
			case egen::TAN:
			case egen::EXP:
			case egen::LOG:
			case egen::SQRT:
			case egen::ROUND:
			case egen::SIGMOID:
			case egen::TANH:
			case egen::SQUARE:
			case egen::CUBE:
			{
				auto var = eteq::make_constant_scalar<float>(
					0.5, teq::Shape({56, 57, 58}));
				auto f = eteq::make_functor<float>(op, {
					eteq::identity_map(var)});
				TIME(f->update())
			}
				break;

			// elementary binary
			case egen::POW:
			case egen::ADD:
			case egen::SUB:
			case egen::MUL:
			case egen::DIV:
			case egen::MIN:
			case egen::MAX:
			case egen::EQ:
			case egen::NEQ:
			case egen::LT:
			case egen::GT:
			case egen::RAND_UNIF:
			{
				auto var = eteq::make_constant_scalar<float>(
					0.5, teq::Shape({56, 57, 58}));
				auto f = eteq::make_functor<float>(op, {
					eteq::identity_map(var), eteq::identity_map(var)});
				TIME(f->update())
			}
				break;

			// reductions
			case egen::REDUCE_SUM:
			case egen::REDUCE_PROD:
			case egen::REDUCE_MIN:
			case egen::REDUCE_MAX:
			{
				auto var = eteq::make_constant_scalar<float>(
					0.5, teq::Shape({56, 57, 58}));
				auto f = eteq::make_functor<float>(op, {
					eteq::reduce_map(var, 1, 1)});
				TIME(f->update())
			}
				break;

			// other stuff
			case egen::PERMUTE:
			{
				auto var = eteq::make_constant_scalar<float>(
					0.5, teq::Shape({56, 57, 58}));
				auto f = eteq::make_functor<float>(op, {
					eteq::permute_map(var, {2, 0, 1})});
				TIME(f->update())
			}
				break;

			case egen::EXTEND:
			{
				auto var = eteq::make_constant_scalar<float>(
					0.5, teq::Shape({56, 58}));
				auto f = eteq::make_functor<float>(op, {
					eteq::extend_map(var, 2, {57})});
				TIME(f->update())
			}
				break;

			case egen::SLICE:
			{
				auto var = eteq::make_constant_scalar<float>(
					0.5, teq::Shape({56, 57, 58}));
				auto f = eteq::make_functor<float>(op, {
					eteq::slice_map(var, {{0,1234},{0,1234},{2,2}})});
				TIME(f->update())
			}
				break;

			case egen::MATMUL:
			{
				auto a = eteq::make_constant_scalar<float>(
					0.3, teq::Shape({253, 255}));
				auto b = eteq::make_constant_scalar<float>(
					0.6, teq::Shape({254, 253}));
				auto f = tenncor::matmul(a, b);
				TIME(f->update())
			}
				break;

			case egen::CONV:
			{
				auto img = eteq::make_constant_scalar<float>(
					0.3, teq::Shape({254, 255}));
				auto kern = eteq::make_constant_scalar<float>(
					0.6, teq::Shape({5, 7}));
				auto f = tenncor::convolution(img, kern, {0, 1});
				TIME(f->update())
			}
				break;

			case egen::PAD:
			{
				auto var = eteq::make_constant_scalar<float>(
					0.5, teq::Shape({56, 57, 58}));
				auto f = eteq::make_functor<float>(op, {
					eteq::pad_map(var, {{0, 0}, {0, 0}, {3, 4}})});
				TIME(f->update())
			}
				break;

			case egen::SELECT:
			{
				teq::Shape shape({56, 57, 58});
				size_t n = shape.n_elems();
				std::vector<float> data;
				data.reserve(n);
				for (size_t i = 0; i < n; ++i)
				{
					data.push_back(i % 2);
				}
				auto cond = eteq::make_constant<float>(data.data(), shape);
				auto a = eteq::make_constant_scalar<float>(0.3, shape);
				auto b = eteq::make_constant_scalar<float>(0.6, shape);
				auto f = tenncor::if_then_else(cond, a, b);
				TIME(f->update())
			}
				break;
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
	opweights.set_label("eteq_weights");
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
