#include "internal/global/g3logs.hpp"

#ifdef GLOBAL_G3LOGGER_HPP

namespace global
{

struct StdioSink
{
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
		else if (g3::internal::wasFatal(level))
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
		if (g3::kThrowErrValue == level.value)
		{
			throw std::runtime_error(msg);
		}
		std::cout << "\033[" << color << "m" << msg << "\033[m" << std::endl;
		// todo: switch back to using default toString once source_location is used
	}
};

void add_stdio_sink (G3WorkerT& worker)
{
	worker->addSink(std::make_unique<StdioSink>(),
		&StdioSink::ReceiveLogMessage);
}

}

#endif
