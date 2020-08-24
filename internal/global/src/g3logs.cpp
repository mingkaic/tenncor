#include "internal/global/g3logs.hpp"

#ifdef GLOBAL_G3LOGGER_HPP

namespace global
{

struct StreamSink
{
	StreamSink (std::ostream& outs) : outs_(outs) {}

	// Linux xterm color
	// http://stackoverflow.com/questions/2616906/how-do-i-output-coloured-text-to-a-linux-terminal
	enum TermColor {YELLOW = 33, RED = 31, GREEN=32, WHITE = 97};

	TermColor get_color (const LEVELS level) const
	{
		if (DBUG.value == level.value)
		{
			return GREEN;
		}
		else if (WARNING.value == level.value)
		{
			return YELLOW;
		}
		else if (WARNING.value < level.value)
		{
			return RED;
		}
		return WHITE;
	}

	void ReceiveLogMessage (g3::LogMessageMover entry)
	{
		auto level = entry.get()._level;
		auto color = get_color(level);
		auto msg = entry.get().toString(
			[](const g3::LogMessage& msg){ return msg.timestamp() + "\t"; });
		outs_ << "\033[" << color << "m" << msg << "\033[m";
		// todo: switch back to using default toString once source_location is used
	}

	std::ostream& outs_;
};

void add_stdio_sink (G3WorkerT& worker, std::ostream& outs)
{
	worker->addSink(std::make_unique<StreamSink>(outs),
		&StreamSink::ReceiveLogMessage);
}

}

#endif
