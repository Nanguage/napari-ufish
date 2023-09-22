from napari_ufish import make_sample_data

# add your tests here...


def test_something():
    data = make_sample_data()
    assert len(data) == 1
    assert data[0][0].shape == (512, 512)
