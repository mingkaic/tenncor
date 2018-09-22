
template <typename T>
struct PrettyTensor
{
	PrettyTensor (std::vector<uint8_t> slist,
		std::vector<uint16_t> show_limit = std::vector<uint16_t>(8, 0x100)) :
		slist_(slist.begin(), slist.end()), show_limit_(show_limit) {}

	void print (std::ostream& out, T* arr)
	{
		print_helper(out, arr, slist_.size() - 1);
	}

private:
	void print_helper (std::ostream& out, T* arr, uint8_t rank)
	{
		out << "[";
		uint16_t n = slist_[rank];
		if (rank < show_limit_.size())
		{
			n = std::min(n, show_limit_[rank]);
		}
		if (rank == 0)
		{
			out << arr[0];
			for (uint16_t i = 1; i < n; ++i)
			{
				out << "," << arr[i];
			}
		}
		else
		{
			auto it = slist_.begin();
			size_t before = std::accumulate(it, it + rank, 1,
				std::multiplies<size_t>());
			print_helper(out, arr, rank - 1);
			for (uint16_t i = 1; i < n; ++i)
			{
				out << ",";
				print_helper(out, arr + i * before, rank - 1);
			}
		}
		if (n < slist_[rank])
		{
			out << "..";
		}
		out << "]";
	}

	std::vector<uint8_t> slist_;

	std::vector<uint16_t> show_limit_;
};
