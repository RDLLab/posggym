
def pytest_addoption(parser):
    parser.addoption(
        "--test_render", action="store_true",
        help="test environment rendering along with other tests (slower)."
    )
    parser.addoption(
        "--env_name_prefix",
        action="store",
        default=None,
        help=(
            "name prefix of environments to test (default is to "
            "test all registered environments)."
        )
    )


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.test_render
    if 'test_render' in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("test_render", [option_value])
    if "env_name_prefix" in metafunc.fixturenames:
        metafunc.parametrize(
            "env_name_prefix", [metafunc.config.getoption("env_name_prefix")]
        )
