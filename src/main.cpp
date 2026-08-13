#include "application.h"

int main(int argc, char* argv[])
{
    CommandLine cmd;

    if (!ParseCommandLine(argc, argv, &cmd))
        return 1;

    Application app(cmd);

    return app.Run();
}