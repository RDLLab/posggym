def pytest_addoption(parser):
    """Add command line options to pytest."""
    parser.addoption(
        "--env_name_prefix",
        action="store",
        default=None,
        help=(
            "name prefix of environments to test (default is to "
            "test all registered environments)."
        ),
    )


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    if "env_name_prefix" in metafunc.fixturenames:
        metafunc.parametrize(
            "env_name_prefix", [metafunc.config.getoption("env_name_prefix")]
        )


# name prefix of environments to test
# Usage: pytest <test files> --env_name_prefix <name>
#
# This limits the environments to be tested to those whose ID starts with <name>.
# Will test all registered environments if not specified.
env_name_prefix = None


def pytest_configure(config):
    """Configure pytest."""
    global env_name_prefix
    env_name_prefix = config.getoption("--env_name_prefix")
