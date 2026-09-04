import pytest

from spnmf.cli import main
from spnmf.io import read_audio, write_audio
from spnmf.signals import percussive_signal, synthetic_mixture


@pytest.fixture
def audio_files(tmp_path):
    mix, _, _, sample_rate = synthetic_mixture(duration=1.5, sample_rate=16000)
    mix_path = tmp_path / "mix.wav"
    drums_path = tmp_path / "drums.wav"
    write_audio(mix_path, mix, sample_rate)
    write_audio(drums_path, percussive_signal(1.5, sample_rate, random_state=1), sample_rate)
    return mix_path, drums_path, sample_rate


def test_separate_writes_both_stems(audio_files, tmp_path, capsys):
    mix_path, _, sample_rate = audio_files
    outdir = tmp_path / "out"
    assert main(
        ["separate", str(mix_path), "-o", str(outdir),
         "--n-fft", "512", "--hop-length", "128", "--n-iter", "10"]
    ) == 0

    harmonic, rate = read_audio(outdir / "mix_harmonic.wav")
    percussive, _ = read_audio(outdir / "mix_percussive.wav")
    assert rate == sample_rate
    assert harmonic.shape == percussive.shape
    assert "wrote" in capsys.readouterr().out


def test_separate_with_a_trained_dictionary(audio_files, tmp_path):
    mix_path, drums_path, _ = audio_files
    outdir = tmp_path / "out"
    assert main(
        ["separate", str(mix_path), "-o", str(outdir), "--dictionary", str(drums_path),
         "--dict-size", "6", "--n-fft", "512", "--hop-length", "128", "--n-iter", "10"]
    ) == 0
    assert (outdir / "mix_harmonic.wav").exists()


def test_quiet_mode_prints_nothing(audio_files, tmp_path, capsys):
    mix_path, _, _ = audio_files
    assert main(
        ["separate", str(mix_path), "-o", str(tmp_path / "out"), "-q",
         "--n-fft", "512", "--hop-length", "128", "--n-iter", "5"]
    ) == 0
    assert capsys.readouterr().out == ""


def test_sample_rate_mismatch_is_reported(audio_files, tmp_path):
    mix_path, _, _ = audio_files
    other = tmp_path / "other_rate.wav"
    write_audio(other, percussive_signal(1.0, 8000, random_state=2), 8000)
    with pytest.raises(SystemExit, match="Hz"):
        main(
            ["separate", str(mix_path), "-o", str(tmp_path / "out"),
             "--dictionary", str(other), "--n-iter", "5"]
        )


def test_demo_runs_and_writes_wavs(tmp_path, capsys):
    outdir = tmp_path / "demo"
    assert main(["demo", "-o", str(outdir), "--duration", "1.0", "--n-iter", "10"]) == 0
    assert len(list(outdir.glob("*.wav"))) > 0
    assert "median HPSS" in capsys.readouterr().out


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "spnmf" in capsys.readouterr().out
