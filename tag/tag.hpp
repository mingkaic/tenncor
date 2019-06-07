#include <set>

#include "ade/ade.hpp"

#ifndef TAG_TAG_HPP
#define TAG_TAG_HPP

namespace tag
{

// each tag instance is a set of tags with particular properties
// iTag is the interface for such instances
struct iTag
{
	virtual ~iTag (void) = default;

	virtual size_t tag_id (void) const = 0;

	virtual void absorb (std::unique_ptr<iTag>&& other) = 0;
};

// TagCollective is a collective of generic iTag instances
// only 1 instance of a particular type of iTag can be stored in an instance of
// TagCollective, adding subsequent instances of the same type
// absorbs new instances into the collective
struct TagCollective final
{
	template <typename TAG>
	static size_t register_tag (void)
	{
		static_assert(std::is_base_of<iTag,TAG>::value);
		const std::type_info& tp = typeid(TAG);
		size_t code = tp.hash_code();
		if (tag_types_.end() == tag_types_.find(code))
		{
			tag_types_.emplace(code);
			return tag_types_.size();
		}
		return 0; // valid tag type ids are greater than 0
	}

	void add (std::unique_ptr<iTag> entry)
	{
		size_t tid = entry->tag_id();
		auto it = tags_.find(tid);
		if (tags_.end() == it)
		{
			tags_.emplace(tid, std::move(entry));
		}
		else
		{
			it->second->absorb(std::move(entry));
		}
	}

private:
	std::unordered_map<size_t,std::unique_ptr<iTag>> tags_;

	static std::unordered_set<size_t> tag_types_;
};

using TensSetT = std::unordered_set<ade::iTensor*>;

// groups are ordered:
// meaning the first label appears before
// subsequent labels obtained from absorption
// e.g.: given tensor X, tag X with A, then tag X wit B,
// results in X having a collective [GroupTag:[A,B]]
struct GroupTag final : public iTag
{
	static std::unordered_map<std::string,TensSetT> groups_;

	GroupTag (std::string init_label) : labels_({init_label}) {}

	size_t tag_id (void) const override
	{
		return tag_id_;
	}

	void absorb (std::unique_ptr<iTag>&& other) override
	{
		std::set<std::string>& olabels =
			static_cast<GroupTag*>(other.get())->labels_;
		labels_.insert(olabels.begin(), olabels.end());
		other.release();
	}

private:
	std::set<std::string> labels_;

	static size_t tag_id_;
};

void group_tag (ade::iTensor* tens, std::string group);

}

#endif // TAG_TAG_HPP
