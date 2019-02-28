#include <fstream>
#include <random>

#include "ead/ead.hpp"

#include "ead/opt/multi_opt.hpp"
#include "ead/opt/ops_reuse.hpp"

#include "dbg/ade_csv.hpp"


static std::random_device rnd_device;
static std::mt19937 mersenne_engine(rnd_device());


static std::vector<double> random_data (size_t n, double lower, double upper)
{
	std::vector<double> out(n);
	std::uniform_real_distribution<double> dist(lower, upper);
	std::generate(out.begin(), out.end(),
		[&dist]() { return dist(mersenne_engine); });
	return out;
}


template <typename T>
struct DebugUpdateSession final : public ade::iTraveler
{
	DebugUpdateSession (ead::Session<T>* sess) : sess_(sess) {}

	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override {}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (nano_durs_.end() == nano_durs_.find(func))
		{
			size_t duration_ns = 0;

			auto& update_set = sess_->need_update_;
			auto& desc_set = sess_->descendants_[func];
			auto has_update =
				[&update_set](ade::iTensor* tens)
				{
					return update_set.end() != update_set.find(tens);
				};
			if (update_set.end() != update_set.find(func) ||
				std::any_of(desc_set.begin(), desc_set.end(), has_update))
			{
				const ade::ArgsT& args = func->get_children();
				for (const ade::FuncArg& arg : args)
				{
					ade::iTensor* tens = arg.get_tensor().get();
					auto& child_desc_set = sess_->descendants_[tens];
					if (update_set.end() != update_set.find(tens) ||
						std::any_of(child_desc_set.begin(),
							child_desc_set.end(), has_update))
					{
						tens->accept(*this);
					}
				}
				// start time
				auto start = std::chrono::steady_clock::now();

				static_cast<ead::Functor<T>*>(func)->update();

				// end time
				auto end = std::chrono::steady_clock::now();
				duration_ns = std::chrono::duration_cast<
				   std::chrono::nanoseconds>(end - start).count();
			}
			nano_durs_.emplace(func, duration_ns);
		}
	}

	std::unordered_map<ade::iTensor*,size_t> nano_durs_;

	ead::Session<T>* sess_;
};

int main (int argc, char** argv)
{
	// optimized sigmoid MLP
	ade::Shape in_shape({10, 3});
	ade::Shape weight0_shape({9, 10});
	ade::Shape bias0_shape({9});
	ade::Shape weight1_shape({5, 9});
	ade::Shape bias1_shape({5});
	ade::Shape out_shape({5,3});

	ead::VarptrT<double> in = ead::make_variable<double>(in_shape, "in");
	ead::VarptrT<double> weight0 = ead::make_variable<double>(weight0_shape, "weight0");
	ead::VarptrT<double> bias0 = ead::make_variable<double>(bias0_shape, "bias0");
	ead::VarptrT<double> weight1 = ead::make_variable<double>(weight1_shape, "weight1");
	ead::VarptrT<double> bias1 = ead::make_variable<double>(bias1_shape, "bias1");
	ead::VarptrT<double> out = ead::make_variable<double>(out_shape, "out");

	ead::NodeptrT<double> intens(in);
	ead::NodeptrT<double> weight0tens(weight0);
	ead::NodeptrT<double> bias0tens(bias0);
	ead::NodeptrT<double> weight1tens(weight1);
	ead::NodeptrT<double> bias1tens(bias1);
	ead::NodeptrT<double> outtens(out);

	auto layer0 = age::add(age::matmul(intens, weight0tens), age::extend(bias0tens, 1, {3}));
	auto sig0 = age::sigmoid(layer0);

	auto layer1 = age::add(age::matmul(sig0, weight1tens), age::extend(bias1tens, 1, {3}));
	auto sig1 = age::sigmoid(layer1);

	auto err = age::pow(age::sub(outtens, sig1),
		ead::make_constant_scalar<double>(2, out_shape));

	auto dw0 = ead::derive(err, weight0tens);
	auto db0 = ead::derive(err, bias0tens);
	auto dw1 = ead::derive(err, weight1tens);
	auto db1 = ead::derive(err, bias1tens);

	// optimize
	auto roots = ead::ops_reuse<double>(ead::multi_optimize<double>({dw0, db0, dw1, db1}));
	dw0 = roots[0];
	db0 = roots[1];
	dw1 = roots[2];
	db1 = roots[3];

	ead::Session<double> session;
	session.track(dw0);
	session.track(db0);
	session.track(dw1);
	session.track(db1);

	std::vector<double> in_data = random_data(in_shape.n_elems(), 0, 1);
	std::vector<double> w0_data = random_data(weight0_shape.n_elems(), 0, 1);
	std::vector<double> b0_data = random_data(bias0_shape.n_elems(), 0, 1);
	std::vector<double> w1_data = random_data(weight1_shape.n_elems(), 0, 1);
	std::vector<double> b1_data = random_data(bias1_shape.n_elems(), 0, 1);
	std::vector<double> out_data = random_data(out_shape.n_elems(), 0, 1);

	in->assign(in_data.data(), in->shape());
	out->assign(out_data.data(), out->shape());
	weight0->assign(w0_data.data(), weight0->shape());
	bias0->assign(b0_data.data(), bias0->shape());
	weight1->assign(w1_data.data(), weight1->shape());
	bias1->assign(b1_data.data(), bias1->shape());
	DebugUpdateSession<double> debugger = session.update<
	DebugUpdateSession<double>>({
		in->get_tensor().get(),
		out->get_tensor().get(),
		weight0->get_tensor().get(),
		bias0->get_tensor().get(),
		weight1->get_tensor().get(),
		bias1->get_tensor().get(),
	});

	CSVEquation ceq;
	ceq.showshape_ = true;
	dw0->get_tensor()->accept(ceq);
	db0->get_tensor()->accept(ceq);
	dw1->get_tensor()->accept(ceq);
	db1->get_tensor()->accept(ceq);

	std::stringstream ss;
	ceq.to_stream(ss);

	std::unordered_map<size_t,ade::iTensor*> idmapper;
	for (auto node : ceq.nodes_)
	{
		idmapper.emplace(node.second.id_, node.first);
	}

	std::ofstream outgraph("/tmp/performance.csv");
	std::string line;
	while (std::getline(ss, line))
	{
		size_t firstsep = line.find(':');
		std::string id(line.begin(), line.begin() + firstsep);
		size_t iid = std::stoi(id);
		auto it = idmapper.find(iid);
		if (idmapper.end() != it)
		{
			size_t nano_dur = debugger.nano_durs_[it->second];
			line = fmts::sprintf("%s,%d", line.c_str(), nano_dur);
		}
		outgraph << line << "\n";
	}

	return 0;
}
