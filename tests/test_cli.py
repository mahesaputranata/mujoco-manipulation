"""Every script's command line must build and parse: a clash between two options with the same
name (render_trials.py's playback --speed vs the scenario's hand --speed) once broke a script
without any other test noticing."""
import pytest


@pytest.mark.parametrize("module", ["eval_place", "render_trials"])
def test_parser_builds_and_parses_defaults(module):
    parser = __import__(module).build_parser()
    args = parser.parse_args([])
    assert args.seed == 0 and args.speed == 0.0  # the scenario's hand speed keeps its name


def test_render_playback_is_separate_from_hand_speed():
    import render_trials

    opts = render_trials.build_parser().parse_args(["--speed", "5", "--playback", "3"])
    assert opts.speed == 5 and opts.playback == 3
