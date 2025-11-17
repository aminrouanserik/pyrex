def test_import_api():
    import pyrex.eccentric_extension

    assert hasattr(pyrex.eccentric_extension, "generate_eccentric_waveform")


def test_generate_waveform_runs_minimal():
    from pyrex.eccentric_extension import generate_eccentric_waveform

    wf = generate_eccentric_waveform(
        approximant="TaylorF2",
        mode=[(2, 2)],
        mass1=20.0,
        mass2=20.0,
        distance=10,
        eccentricity=0,
        delta_t=1 / 4096,
        f_lower=20,
    )
    assert wf is not None
    assert hasattr(wf, "time")
    assert hasattr(wf, "strain")
