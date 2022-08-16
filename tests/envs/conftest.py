
def pytest_addoption(parser):
    parser.addoption(
        "--test_render", action="store_true",
        help="test environment rendering along with other tests (slower)."
    )


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.test_render
    if 'test_render' in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("test_render", [option_value])
