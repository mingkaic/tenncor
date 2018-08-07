#include "sand/inode.hpp"

#include "util/error.hpp"
#include "util/rand.hpp"

#ifndef GLASS_SESSION_HPP
#define GLASS_SESSION_HPP

using NamedNodes = std::unordered_map<std::string,std::weak_ptr<iNode> >;

struct Session final
{
	void add (std::string label, Nodeptr& node);

	std::string name (Nodeptr& node) const;

	iNode* get (std::string key) const;

	void clean (void);

	std::string hash (void) const;

	NamedNodes nodes_;

private:
	std::unordered_map<iNode*,std::string> names_;
	std::string uid_ = make_uid(this);
};

extern Session session;

#define SESS(var) session.add(#var, var);

#endif /* GLASS_SESSION_HPP */
